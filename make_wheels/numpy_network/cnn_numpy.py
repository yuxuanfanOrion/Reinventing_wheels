import numpy as np

'''
    Convolutional Neural Network implemented from scratch using numpy

    Network Process:
        1. Input the data into the neural network
        2. Data flows from one layer to another layer until we get the output
        3. Compute the loss, a scalar
        4. Backpropagate the loss to each layer *(Most Important)
        5. Update the weights and biases using the gradients
        6. Iterate until the loss is minimized

    Codes logic:
        1. Network and Architecture initialization
        2. Forward pass
        3. Compute loss
        4. Backward pass
        5. Update weights and biases

    How to build every layer?
        X -> Layer -> Y -> Loss -> Gradients -> Weights and Biases -> Update

    Class List:
    - Layer
    - FC
    - ActivationLayer
    - ConvolutionLayer
        

    For more info, contact me at:
    orionisfan@outlook.com


'''

from layer import Layer
from activation import ActivationLayer


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []
        output = input_data
        
        for i in range(samples):
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
                result.append(output)
        return result
    
    def train(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                # calculate loss
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples
            print('epoch %d, error=%f' % (i, err))