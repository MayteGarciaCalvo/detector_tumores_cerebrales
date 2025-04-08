import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Título de la app
st.title("🧠 Detección de Tumores Cerebrales")
st.write("Sube una imagen de resonancia magnética para predecir si tiene tumor o no. La imagen debe ser de una resonancia cerebral.")

# Cargar el modelo
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("modelo_tumores.h5")
    return model

model = load_model()

# Preprocesamiento de la imagen
def preprocess_image(image):
    image = image.resize((128, 128))
    image = image.convert("RGB")
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Verificar si la imagen es una resonancia magnética
def is_mri(image):
    # Comprobar si la imagen tiene un tamaño razonable (resolución mínima de una resonancia)
    min_resolution = 100  # Resolución mínima, ajustable según el caso
    if image.size[0] < min_resolution or image.size[1] < min_resolution:
        return False  # Si la imagen es demasiado pequeña, no es una resonancia válida
    
    # Comprobar el formato de la imagen (JPEG/PNG)
    if image.format not in ['JPEG', 'PNG']:
        return False  # Filtra imágenes no adecuadas como monedas u otros objetos

    return True

# Subir imagen
uploaded_file = st.file_uploader("📤 Sube una imagen de resonancia cerebral (JPG o PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Abrimos la imagen
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_container_width=True)

    # Verificar si la imagen parece una resonancia magnética
    if not is_mri(image):
        st.warning("⚠️ La imagen no parece ser una resonancia magnética. Por favor, sube una imagen válida de resonancia cerebral.")
    else:
        # Botón para analizar la imagen
        if st.button("🔍 Analizar"):
            input_image = preprocess_image(image)
            prediction = model.predict(input_image)[0][0]

            st.subheader("📊 Resultado:")
            if prediction > 0.5:
                st.error(f"🚨 Tumor detectado (probabilidad: {prediction:.2f})")
            else:
                st.success(f"✅ No se detecta tumor (probabilidad: {1 - prediction:.2f})")

            # Mostrar un poco más de información
            if prediction > 0.5:
                st.markdown("""
                ### ¿Qué significa esto?
                El modelo ha detectado la presencia de un tumor en la imagen. Te recomendamos que contactes con un médico para un diagnóstico profesional.
                """)
            else:
                st.markdown("""
                ### ¿Qué significa esto?
                No se ha detectado un tumor, pero siempre es mejor consultar con un especialista para confirmar el diagnóstico.
                """)

