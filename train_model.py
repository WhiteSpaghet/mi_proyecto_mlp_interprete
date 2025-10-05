from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1️⃣ Cargar MNIST
(x_train, y_train), (_, _) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0
y_train = to_categorical(y_train)

# 2️⃣ Definir el modelo
model = Sequential([
    Dense(256, activation='relu', input_shape=(28*28,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 3️⃣ Compilar y entrenar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 4️⃣ Guardar el modelo en app_web/
model.save("app_web/mnist_model.h5")
print("Modelo guardado en app_web/mnist_model.h5 ✅")
