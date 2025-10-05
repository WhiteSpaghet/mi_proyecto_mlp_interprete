# ========================================
# CELDA 1: Imports y utilidades
# ========================================
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Funciones de activación para MLP NumPy
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# ========================================
# CELDA 2: MLP desde cero con NumPy
# ========================================
class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation(z)

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

# Ejemplo de uso
mlp_numpy = MLP([2, 3, 1], [relu, sigmoid])
X_test = np.array([[0.5, -0.2]])
print("Output MLP NumPy:", mlp_numpy.predict(X_test))
# ========================================
# CELDA 3: Intérprete de arquitecturas
# ========================================
def compile_model(architecture_string, input_shape=(28*28,)):
    model = tf.keras.Sequential()
    layers = architecture_string.split("->")
    
    first = True
    for layer_str in layers:
        layer_str = layer_str.strip()
        type_start = layer_str.find("(")
        type_end = layer_str.find(")")
        layer_type = layer_str[:type_start]
        params = layer_str[type_start+1:type_end].split(",")
        units = int(params[0].strip())
        activation = params[1].strip() if len(params) > 1 else None

        if first:
            model.add(tf.keras.layers.Dense(units, activation=activation, input_shape=input_shape))
            first = False
        else:
            model.add(tf.keras.layers.Dense(units, activation=activation))
    
    return model

# Ejemplo de compilación
example_architecture = "Dense(16, relu) -> Dense(8, relu) -> Dense(10, softmax)"
model_example = compile_model(example_architecture, input_shape=(5,))
print("Resumen modelo compilado:")
model_example.summary()
# ========================================
# CELDA 4: Cargar y preparar MNIST
# ========================================
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Shape X_train:", x_train.shape)
print("Shape y_train:", y_train.shape)
# ========================================
# CELDA 5: Entrenamiento del modelo compilado
# ========================================
# Define la arquitectura que quieras entrenar
architecture = "Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
model = compile_model(architecture)

# Configurar y entrenar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
# ========================================
# CELDA 6: Evaluación
# ========================================
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Precisión en test:", test_acc)
# ========================================
# CELDA 7: Gráficas de evolución
# ========================================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()