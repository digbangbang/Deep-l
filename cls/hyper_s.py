from train import train, evaluate
from model import NeuralNetwork

def hyperparameter_search():
    hidden_sizes = [64, 128, 256]
    learning_rates = [0.4, 0.35, 0.3]
    regularization_strengths = [0.0, 0.01]
    best_accuracy = 0
    best_params = {}
    
    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for reg_strength in regularization_strengths:
                print(f'Training with hidden_size={hidden_size}, learning_rate={learning_rate}, reg_strength={reg_strength}')
                model = NeuralNetwork(input_size, hidden_size, output_size)
                train(model, x_train, y_train, x_val, y_val, epochs, learning_rate, reg_strength, x_test, y_test)
                accuracy = evaluate(model, x_test, y_test)
                print(f'Accuracy: {accuracy * 100:.2f}%')
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {
                        'hidden_size': hidden_size,
                        'learning_rate': learning_rate,
                        'reg_strength': reg_strength
                    }
    
    print(f'Best params: {best_params}')
    print(f'Best accuracy: {best_accuracy * 100:.2f}%')