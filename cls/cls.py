import numpy as np

from data import load_data, train_val_split
from train import train, evaluate
from model import NeuralNetwork
from save import save_weights
x_train, y_train, x_test, y_test = load_data()
x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)

input_size = 28 * 28
hidden_size = 128
output_size = 10
epochs = 500
learning_rate = 0.3
regularization_strength = 0.01

model = NeuralNetwork(input_size, hidden_size, output_size)
train(model, x_train, y_train, x_val, y_val, epochs, learning_rate, regularization_strength, x_test, y_test)
accuracy = evaluate(model, x_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

save_weights('best_model.npz', model)

# hyperparameter_search()
