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

#! Base Class for all layers
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    # Compute the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # Compute dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, loss, learning_rate):
        raise NotImplementedError

class FC(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, loss, learning_rate):
        loss = np.dot(loss, self.weights.T)
        weights_loss = np.dot(self.input.T, loss)
        self.weights -= learning_rate * weights_loss
        self.bias -= learning_rate * loss
        return loss
    pass


#! Activation Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, loss, learning_rate):
        return self.activation_prime(self.input) * loss
    pass


    