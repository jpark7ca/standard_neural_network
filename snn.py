import random
import numpy as np
import os
import pickle
import sys
import matplotlib.pyplot as plt
import zipfile as zip

#######################################
# Global functions
#######################################
def rescale_pixel(a):
   return a/255

#######################################
# Cost function class
#######################################
class Cost:
    @staticmethod
    def loss(type):
        if type == "crossentropy":
            return Cost.crossentropy
        else:
            return None 

    @staticmethod
    def z_loss(type):
        if type == "crossentropy":
            return Cost.z_crossentropy
        else:
            return None

    @staticmethod
    def crossentropy(type, a, y):
        costv = None
        if type == "sigmoid":
            costv = np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a))
        if type == "softmax":
            costv = np.nan_to_num(-y*np.log(a))
        
        _num_data = costv.shape[0]
        return costv.sum(axis=1).sum(axis=0) / _num_data
        
    @staticmethod
    def z_crossentropy(type, a, y):
        dcdz = None
        if type == "sigmoid":
            dcdz = (a-y)
        if type == "softmax":
            dcdz = (a-y)
        
        return dcdz

#######################################
# Activation class
#######################################
class Activation:
    @staticmethod
    def a(type):
        if type == "sigmoid":
            return Activation.sigmoid
        elif type == "tanh":
            return Activation.tanh
        elif type == "softmax":
            return Activation.softmax
        else:
            return None

    @staticmethod
    def d_a(type):
        if type == "sigmoid":
            return Activation.d_sigmoid
        elif type == "tanh":
            return Activation.d_tanh
        else:
            return None

    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0 + np.exp(-z))

    @staticmethod
    def d_sigmoid(z):
        return Activation.sigmoid(z)*(1-Activation.sigmoid(z))
    
    @staticmethod
    def softmax(z):
        _sum = np.expand_dims(np.exp(z).sum(axis=1), axis=2)
        return np.exp(z) / _sum

    @staticmethod
    def tanh(z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    
    @staticmethod
    def d_tanh(z):
        return 1 - np.power(Activation.tanh(z), 2)

#######################################
# Layer class
#######################################
class FullyConnectedLayer:
    def __init__(self, neurons, input=None, batch=None, activation=None):
        self.outputsize = neurons       # number of neurons 
        self.inputsize = input          # number of input
        self.batchsize = batch          # batch size 
        self.activation = activation
        self.z = None
        self.a = None
        self.g_weights = None
        self.g_biases = None

        # Initialize transposed(*) weights and biases if input defined.  
        self.weights = None if self.inputsize is None else np.random.randn(1, self.outputsize, self.inputsize)
        self.biases = None if self.inputsize is None else np.random.randn(1, self.outputsize, 1)

    def configure(self, batch):
        # Set a batch size
        self.batchsize = batch
        
        self.z = None if self.inputsize is None else np.zeros(((self.batchsize if self.batchsize is not None else 1), self.outputsize, 1))
        self.a = np.zeros(((self.batchsize if self.batchsize is not None else 1), self.outputsize, 1))

        self.g_weights = None if self.inputsize is None else np.random.randn((self.batchsize if self.batchsize is not None else 1), self.outputsize, self.inputsize)
        self.g_biases = None if self.inputsize is None else np.random.randn((self.batchsize if self.batchsize is not None else 1), self.outputsize, 1)
        

############################################
# Standard Neural Network (network) class
############################################
class StandardNeuralNetwork:
    
    def __init__(self, network_layers_description=None, cost=None):
        self.network_layers_description = network_layers_description
        self.layers = []
        self.cost = "crossentropy" if cost is None else cost # default to crossentropy
        num_input = None

        if(self.network_layers_description is not None):
            for l in self.network_layers_description:
                if(l[0] == "FullyConnected"):
                    self.add(FullyConnectedLayer(neurons=l[1], input=num_input, batch=None, activation=l[2]))
                    num_input = l[1]

    def evaluate(self, input, debug):
        batch = input[0].shape[0]
        last_layer_index = len(self.layers) - 1
        for l in self.layers:
            l.configure(batch=batch)
            
        a = input[0]
        y = input[1] # one-hot encodeding labels (1 or 0)

        for index, layer in enumerate(self.layers):
            if(layer.weights is None):          # input layer
                layer.a = a
            elif index < last_layer_index:      # layers except the last one
                a = Activation.a(layer.activation)(np.matmul(layer.weights, a) + layer.biases)
                layer.a = a
            else:                               # last layer 
                a = Activation.a(layer.activation)(np.matmul(layer.weights, a) + layer.biases)
                layer.a = a
                a1 = np.argmax(layer.a, axis=1)
                y1 = np.argmax(y, axis=1)
                s1 = np.sum(np.where(a1 == y1, 1, 0))

        return s1

    def execute(self, input, epochs, eta, _lambda, batch, input_test=None, debug=None):
        num_data = input[0].shape[0]
        idx = []

        for i in xrange(0, num_data):
            idx.append(i)
        
        # epochs
        for e in xrange(0, epochs):
            print "epoch {0} started".format(e)
            # shuffle
            random.shuffle(idx)
            
            idxbatches = [
                idx[i:i+batch] for i in xrange(0, num_data, batch)
            ]

            for index, idxbatch in enumerate(idxbatches):                
                inputbatch = None
                labelbatch = None 
                for i in idxbatch:
                    if inputbatch is None:
                        inputbatch = input[0][i:i+1, 0:, 0:]
                    else:
                        inputbatch = np.append(inputbatch, input[0][i:i+1, 0:, 0:], axis=0)
                    
                    if labelbatch is None:
                        labelbatch = input[1][i:i+1, 0:, 0:]
                    else:
                        labelbatch = np.append(labelbatch, input[1][i:i+1, 0:, 0:], axis=0)
                
                self.run_batch((inputbatch, labelbatch), eta, _lambda, num_data, debug)
            
            print "Epoch {0} complete".format(e)

            # check accuracy
            if(input_test):
                match = self.evaluate(input_test, debug)
                total = input_test[0].shape[0]
                print "Epoch {0}: Total Match: {1} Total Test Data: {2} ( {3}% )".format(e, match, total, (float(match)/float(total))*100.)

    def run_batch(self, input, eta, _lambda, num_data, debug=None):
        batch = input[0].shape[0]

        for l in self.layers:
            l.configure(batch=batch)
            
        a = input[0]
        y = input[1]    # one-hot encodeding labels (1 or 0)

        ########################################################
        ### FEED FORDWARD
        ########################################################

        for index, layer in enumerate(self.layers):
            if(layer.weights is None):  
                layer.a = a
            else:
                layer.z = np.matmul(layer.weights, a) + layer.biases
                a = Activation.a(layer.activation)(layer.z)
                layer.a = a

        ########################################################
        ### BACKPROPGATION
        ########################################################
        
        for i in xrange(len(self.layers)-1, 0, -1):
            
            if(i == len(self.layers)-1):
                prevdz = Cost.z_loss(self.cost)(self.layers[i].activation, self.layers[i].a, y)
            else:
                prevda = np.matmul(np.transpose(self.layers[i+1].weights, (0, 2, 1)), prevdz)   
                prevdz = prevda * Activation.d_a(self.layers[i].activation)(self.layers[i].z) 

            self.layers[i].g_biases = prevdz
            self.layers[i].g_weights = np.matmul(prevdz, np.transpose(self.layers[i-1].a, (0, 2, 1)))
            
        ########################################################
        ### UPDATE weights and biases
        ########################################################

        for i in xrange(len(self.layers)-1, 0, -1):
            
            c_g_biases = self.layers[i].g_biases.sum(axis=0, keepdims=True)
            c_g_weights = self.layers[i].g_weights.sum(axis=0, keepdims=True) 

            self.layers[i].biases = self.layers[i].biases - (eta/self.layers[i].batchsize) * c_g_biases
            self.layers[i].weights = (1-(eta*(_lambda/num_data))) * self.layers[i].weights - (eta/self.layers[i].batchsize) * c_g_weights

    def add(self, layer):
        self.layers.append(layer)
    
    def describe_layers(self):
        print "[Standard Neural Network]"
        print "Number of layers: {0}".format(len(self.layers)) 
        print "Cost function: {0}".format(self.cost)
        for index, layer in enumerate(self.layers):
            print "==========================================="
            print "Layer {0}".format(index)
            print "Number of inputs: {0}".format(layer.inputsize)
            print "Number of neurons (outputs): {0}".format(layer.outputsize)
            print "Batch size: {0}".format(layer.batchsize)
            print "Weights: {0}".format(None if layer.weights is None else layer.weights.shape)
            print "Biases: {0}".format(None if layer.biases is None else layer.biases.shape)
            print "z: {0}".format(None if layer.z is None else layer.z.shape)
            print "a={1}(z): {0}".format(None if layer.a is None else layer.a.shape, layer.activation)
            print "Gradient Weights: {0}".format(None if layer.g_weights is None else layer.g_weights.shape)
            print "Gradient Biases: {0}".format(None if layer.g_biases is None else layer.g_biases.shape)
            print "Activation: {0}".format(layer.activation)
    

############################################
# Main routine
############################################
print "Unziping training dataset ..."
zip = zip.ZipFile('mnist_train_pkl.zip')
zip.extractall()

print "Building a standard neural network ..."
snn = StandardNeuralNetwork(network_layers_description=[["FullyConnected", 784, None], ["FullyConnected", 30, "sigmoid"], ["FullyConnected", 10, "softmax"]], cost="crossentropy")

print "Loading data set ..."
rfile = open("mnist_train.pkl", "rb")
mnist = pickle.load(rfile)
rfile.close()
inputs = (rescale_pixel((mnist["inputs"][0:50000]).astype('float32')), mnist["labels"][0:50000]) 
inputs_test = (rescale_pixel((mnist["inputs"][50000:60000]).astype('float32')), mnist["labels"][50000:60000])

print "Running training sessions..."
snn.execute(input=inputs, epochs=10, eta=3.0, _lambda=0, input_test=inputs_test, batch=50, debug=False)

print "==============================================="
snn.describe_layers()