import numpy as np

from activation_f import relu, relu_derivative, softmax

# class NeuralNetwork:
#     def __init__(self, input_size, hidden_size, output_size):
#         self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
#         self.bias1 = np.zeros((1, hidden_size))
#         self.weights2 = np.random.randn(hidden_size, int(1.5 * hidden_size)) * 0.01
#         self.bias2 = np.zeros((1, int(1.5 * hidden_size)))
#         self.weights3 = np.random.randn(int(1.5 * hidden_size), output_size) * 0.01
#         self.bias3 = np.zeros((1, output_size))
        
#     def forward(self, x):
#         self.z1 = np.dot(x, self.weights1) + self.bias1
#         self.a1 = relu(self.z1)
#         self.z2 = np.dot(self.a1, self.weights2) + self.bias2
#         self.a2 = relu(self.z2)
#         self.z3 = np.dot(self.a2, self.weights3) + self.bias3
#         self.a3 = softmax(self.z3)
#         return self.a3

#     def backward(self, x, y, output, learning_rate, regularization_strength):
#         m = y.shape[0]
        
#         dz3 = output - y
#         dw3 = np.dot(self.a2.T, dz3) / m + regularization_strength * self.weights3 / m
#         db3 = np.sum(dz3, axis=0, keepdims=True) / m
        
#         dz2 = np.dot(dz3, self.weights3.T) * relu_derivative(self.z2)
#         dw2 = np.dot(self.a1.T, dz2) / m + regularization_strength * self.weights2 / m
#         db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
#         dz1 = np.dot(dz2, self.weights2.T) * relu_derivative(self.z1)
#         dw1 = np.dot(x.T, dz1) / m + regularization_strength * self.weights1 / m
#         db1 = np.sum(dz1, axis=0, keepdims=True) / m

#         self.weights3 -= learning_rate * dw3
#         self.bias3 -= learning_rate * db3
#         self.weights2 -= learning_rate * dw2
#         self.bias2 -= learning_rate * db2
#         self.weights1 -= learning_rate * dw1
#         self.bias1 -= learning_rate * db1

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias2 = np.zeros((1, output_size))
        
    def forward(self, x):
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = softmax(self.z2)
        return self.a2
    
    def backward(self, x, y, output, learning_rate, regularization_strength):
        m = y.shape[0]
        dz2 = output - y
        dw2 = np.dot(self.a1.T, dz2) / m + regularization_strength * self.weights2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = np.dot(dz2, self.weights2.T) * relu_derivative(self.z1)
        dw1 = np.dot(x.T, dz1) / m + regularization_strength * self.weights1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.weights2 -= learning_rate * dw2
        self.bias2 -= learning_rate * db2
        self.weights1 -= learning_rate * dw1
        self.bias1 -= learning_rate * db1

# class NeuralNetwork:
#     def __init__(self, input_size, hidden_size, output_size):
#         self.weights1 = np.random.randn(input_size, output_size) * 0.01
#         self.bias1 = np.zeros((1, output_size))
        
#     def forward(self, x):
#         self.z1 = np.dot(x, self.weights1) + self.bias1
#         self.a1 = softmax(self.z1)
#         return self.a1
    
#     def backward(self, x, y, output, learning_rate, regularization_strength):
#         m = y.shape[0]
#         dz1 = output - y
#         dw1 = np.dot(x.T, dz1) / m + regularization_strength * self.weights1 / m
#         db1 = np.sum(dz1, axis=0, keepdims=True) / m

#         self.weights1 -= learning_rate * dw1
#         self.bias1 -= learning_rate * db1
