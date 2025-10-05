# ========================================
# 1️⃣ Imports y carga de modelos
# ========================================
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

st.title("🖌️ Dibuja un número y predícelo")

# Cargar modelo principal
model_path = os.path.join(os.path.dirname(__file__), "mnist_model.h5")
model = load_model(model_path)

# Cargar un segundo modelo opcional (para comparar)
# model_path2 = os.path.join(os.path.dirname(__file__), "mnist_model2.h5")
# model2 = load_model(model_path2)
# Para usar el segundo modelo, reemplaza 'model' por 'model2' más abajo

# ========================================
# 2️⃣ Canvas para dibujar
# ========================================
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

# ========================================
# 3️⃣ Procesamiento de la imagen
# ========================================
if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype('uint8')
    img = 255 - img[:, :, 0]  # invertir: fondo blanco -> negro para el modelo

    # Recortar zona con dibujo
    coords = np.argwhere(img < 255)
    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img = img[y0:y1+1, x0:x1+1]
        img = Image.fromarray(img).resize((28,28))
    else:
        img = Image.fromarray(np.ones((28,28))*255)  # imagen blanca si no dibujas

    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,28*28)

    # ========================================
    # 4️⃣ Botón de predicción
    # ========================================
    if st.button("Predecir"):
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        st.write(f"Predicción: **{digit}**")
