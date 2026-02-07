import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

MODEL_PATH = '/content/drive/MyDrive/teeth_classification/outputs/models/week2/mobilenetv2_best.keras' 
IMG_SIZE = (224, 224)
CLASS_NAMES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT'] 

# --- UI Setup ---
st.set_page_config(page_title="Teeth Classifier", page_icon="🦷")
st.title("🦷 Teeth Classification AI")
st.write("Upload a dental image to classify it into one of 7 categories.")

# --- Sidebar ---
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Transfer Learning model (MobileNetV2) "
    "trained on the Teeth Dataset to classify dental conditions."
)

# --- Load Model ---
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}.")
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Main Interface ---
file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner("Loading AI Model (this takes ~30s first time)..."):
        model = load_model()
    if model is not None:
        st.write("Analyzing...")
        image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
        image = image.convert("RGB") 
        img_array = np.expand_dims(np.array(image), axis=0)
        
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[class_idx]
        confidence = np.max(predictions[0])
        
        st.success(f"**Prediction:** {predicted_class}")
        st.info(f"**Confidence:** {confidence * 100:.2f}%")
        st.bar_chart(dict(zip(CLASS_NAMES, predictions[0])))






