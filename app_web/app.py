import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.title("🖌️ Dibuja un número y predícelo")

# Cargar modelo
import os
model_path = os.path.join(os.path.dirname(__file__), "mnist_model.h5")
model = load_model(model_path)

# Configuración del canvas
canvas_result = st_canvas(
    fill_color="#FFFFFF",  # fondo blanco
    stroke_width=15,
    stroke_color="#000000",  # trazo negro
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Cuando hay dibujo
if canvas_result.image_data is not None:
    # Convertir imagen a 28x28 y escala de grises
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
    img = img.resize((28,28))
    img_array = np.array(img)/255.0
    img_array = 1 - img_array  # invertir blanco-negro
    img_array = img_array.reshape(1,28*28)

    if st.button("Predecir"):
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        st.write(f"Predicción: **{digit}**")

