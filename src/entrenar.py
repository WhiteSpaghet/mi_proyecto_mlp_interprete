from interprete import compile_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Cargar y preparar MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Arquitectura
architecture = "Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)"
model = compile_model(architecture)

# Entrenar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# Guardar modelo
model.save('mnist_model.h5')
print("Modelo guardado en mnist_model.h5")