import numpy as np
from datasets import load_dataset

def load_data():
    dataset = load_dataset('fashion_mnist')
    x_train = np.array(dataset['train']['image']).reshape(-1, 28*28).astype(np.float32) / 255
    y_train = np.eye(10)[np.array(dataset['train']['label'])]
    x_test = np.array(dataset['test']['image']).reshape(-1, 28*28).astype(np.float32) / 255
    y_test = np.eye(10)[np.array(dataset['test']['label'])]
    return x_train, y_train, x_test, y_test

def train_val_split(x, y, val_ratio=0.2):
    indices = np.random.permutation(len(x))
    val_size = int(len(x) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return x[train_indices], y[train_indices], x[val_indices], y[val_indices]