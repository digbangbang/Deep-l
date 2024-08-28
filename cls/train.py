# import wandb
import numpy as np

from tqdm import tqdm
from activation_f import cross_entropy_loss

# wandb.init(project='FMNIST', name='fmnist', entity='2623448751')

def train(model, x_train, y_train, x_val, y_val, epochs, learning_rate, regularization_strength, x_test, y_test):
    best_loss = float('inf')
    best_weights = model.weights1, model.bias1, model.weights2, model.bias2
    
    lr = learning_rate
    for epoch in tqdm(range(epochs)):
        output = model.forward(x_train)
        loss = cross_entropy_loss(output, y_train)
        model.backward(x_train, y_train, output, lr, regularization_strength)

        lr = 0.1 + 0.5 * (learning_rate - 0.1) * (1 + np.cos(np.pi * epoch / epochs))

        val_output = model.forward(x_val)
        val_loss = cross_entropy_loss(val_output, y_val)
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = model.weights1.copy(), model.bias1.copy(), model.weights2.copy(), model.bias2.copy()
        
        print(f'Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Validation loss: {val_loss:.4f}')
        # accuracy = evaluate(model, x_test, y_test)
        # wandb.log({
        #     'train loss': loss,
        #     'val loss': val_loss,
        #     'test acc': accuracy,
        #     # 'lr': lr
        # })
    
    model.weights1, model.bias1, model.weights2, model.bias2 = best_weights

def evaluate(model, x_test, y_test):
    output = model.forward(x_test)
    predictions = np.argmax(output, axis=1)
    targets = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == targets)
    return accuracy