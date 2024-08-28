import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_shifted = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    return -np.mean(np.sum(targets * np.log(predictions + 1e-9), axis=1))
