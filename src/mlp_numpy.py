import numpy as np

# Funciones de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Clase Layer
class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation(z)

# Clase MLP
class MLP:
    def __init__(self, layer_sizes, activations):
        self.layers = []
        for i in range(len(layer_sizes)-1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i]))

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output