import numpy as np


#! Mean Squared Error
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))
def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size


#! Cross Entropy
def cross_entropy(y_true, y_pred):
    return -np.mean(y_true*np.log(y_pred))

def cross_entropy_prime(y_true, y_pred):
    return -y_true/y_pred


#! Binary Cross Entropy
def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true*np.log(y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return -y_true/y_pred


#! Categorical Cross Entropy
def categorical_cross_entropy(y_true, y_pred):
    return -np.mean(y_true*np.log(y_pred))

def categorical_cross_entropy_prime(y_true, y_pred):
    return -y_true/y_pred

