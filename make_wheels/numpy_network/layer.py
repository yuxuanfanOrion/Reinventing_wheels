import numpy as np

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
    

class FCLayer(Layer):
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



