import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

st.title("🖌️ Reconocimiento de dígitos MNIST")

# Cargar modelo
model = load_model("../src/mnist_model.h5")

# Crear canvas para dibujar
canvas_size = 280
st.write("Dibuja un número (0-9) en el recuadro:")
canvas_image = Image.new("L", (canvas_size, canvas_size), color=255)
draw = ImageDraw.Draw(canvas_image)

# Streamlit simple con botón de limpiar y predecir
if 'canvas_image' not in st.session_state:
    st.session_state.canvas_image = canvas_image

if st.button("Limpiar"):
    st.session_state.canvas_image = Image.new("L", (canvas_size, canvas_size), color=255)

# Mostrar canvas
st.image(st.session_state.canvas_image, caption="Tu dibujo", use_column_width=False)

# Predecir
if st.button("Predecir"):
    # Redimensionar a 28x28 y normalizar
    img = st.session_state.canvas_image.resize((28,28))
    img_array = np.array(img)/255.0
    img_array = 1 - img_array  # invertir blanco-negro
    img_array = img_array.reshape(1,28*28)
    
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    st.write(f"Predicción: **{digit}**")
