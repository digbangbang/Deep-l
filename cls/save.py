import numpy as np

def save_weights(filename, model):
    np.savez(filename, 
             weights1=model.weights1, bias1=model.bias1, 
             weights2=model.weights2, bias2=model.bias2)