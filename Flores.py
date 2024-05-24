import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from io import BytesIO

# Ocultar advertencias de deprecación
import warnings
warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(
    page_title="Reconocimiento de Productos",
    page_icon="icono.png",
    initial_sidebar_state='auto'
)

# Estilo personalizado para ocultar el menú y el pie de página
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./modelo_productos_IA.h5')
    return model
with st.spinner('Modelo está cargando...'):
    model = load_model()

with st.sidebar:
    st.image('foto.jpg')
    st.title("Reconocimiento de imagen")
    st.subheader("Reconocimiento de imagen para Productos")

st.image('logo.png')
st.title("Pruebas")
st.write("Sólo yo, nadie más, quizás Iván, pero nadie más")
st.write("""
         # Detección de Productos
         """)

def import_and_predict(image_data, model, class_names):
    image_data = image_data.resize((180, 180))
    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0)  # Crear un batch

    # Predecir con el modelo
    prediction = model.predict(image)
    index = np.argmax(prediction)
    score = tf.nn.softmax(prediction[0])
    class_name = class_names[index].strip()
    
    return class_name, score

class_names = open("./clases.txt", "r").readlines()

# Opciones para ingresar la imagen
option = st.selectbox(
    'Selecciona cómo deseas subir la imagen:',
    ('Tomar una foto', 'Subir una foto', 'Ingresar URL de una foto')
)

image = None

if option == 'Tomar una foto':
    img_file_buffer = st.camera_input("Capture una foto para identificar una flor")
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)

elif option == 'Subir una foto':
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif option == 'Ingresar URL de una foto':
    url = st.text_input("Ingresa el URL de la imagen")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
        except:
            st.error("No se pudo cargar la imagen desde el URL proporcionado.")

# Procesar la imagen y mostrar resultados
if image is not None:
    st.image(image, use_column_width=True)
    
    # Realizar la predicción
    class_name, score = import_and_predict(image, model, class_names)
    
    # Mostrar el resultado
    if np.max(score) > 0.5:
        st.subheader(f"Producto: {class_name}")
        st.text(f"Puntuación de confianza: {100 * np.max(score):.2f}%")
    else:
        st.text(f"No se pudo determinar el producto")
else:
    st.text("Por favor proporciona una imagen")
