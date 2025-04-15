
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Título de la app
st.title("🧠 Detección de Tumores Cerebrales")
st.write("Sube una imagen de resonancia magnética para predecir si tiene tumor, no tiene tumor o si no es una resonancia válida.")

# Cargar el modelo
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("modelo_tumores_actualizado1.h5")
    return model

model = load_model()

# Preprocesamiento de la imagen
def preprocess_image(image):
    image = image.resize((128, 128))
    image = image.convert("RGB")
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Diccionario de clases
class_names = ["Tumor", "No Tumor", "No Escáner"]

# Subir imagen
uploaded_file = st.file_uploader("📤 Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Imagen cargada", use_container_width =True)

    # Botón para analizar
    if st.button("🔍 Analizar"):
        input_image = preprocess_image(image)
        prediction = model.predict(input_image)[0]
        predicted_class = np.argmax(prediction)
        probability = prediction[predicted_class]

        st.subheader("📊 Resultado del análisis:")
        
        if predicted_class == 0:
            st.error(f"⚠️ Imagen no válida como resonancia (probabilidad: {probability:.2f})")
            st.markdown("### ¿Qué significa esto?\nLa imagen subida **no parece ser una resonancia magnética cerebral**. Asegúrate de subir una imagen válida del cerebro.")
        elif predicted_class == 1:
            st.success(f"🚨 Tumor detectado (probabilidad: {probability:.2f})")
            st.markdown("### ¿Qué significa esto?\nEl modelo ha detectado la **presencia de un tumor** en la imagen. Por favor, contacta con un especialista para una evaluación profesional.")
        else:
            st.warning(f"✅ No se detecta tumor (probabilidad: {probability:.2f})")
            st.markdown("### ¿Qué significa esto?\nNo se ha detectado un tumor en la imagen. Aun así, se recomienda consultar con un médico para confirmar.")
