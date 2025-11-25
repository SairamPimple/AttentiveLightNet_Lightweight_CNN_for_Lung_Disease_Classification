import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os

# Import custom objects for registration
from model_architecture import swish_plus, DynamicDropout, CustomFocalLoss

# --- Configuration ---
MODEL_PATH = 'models/attentive_lightnet.keras'
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Normal', 'Pneumonia'] 

# --- Load Model ---
@st.cache_resource
def load_app_model():
    """Loads the trained Keras model with caching."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None
        
    try:
        # Custom objects are handled via @register_keras_serializable decorators
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def enhance_and_prepare(image_pil):
    """
    Preprocesses PIL image: Grayscale -> Resize -> CLAHE -> Normalize
    Matches training preprocessing exactly.
    """
    # Convert PIL to OpenCV (numpy)
    image_np = np.array(image_pil)
    
    # Convert to Grayscale
    if len(image_np.shape) == 3:
        if image_np.shape[2] == 4: # RGBA
             image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2GRAY)
        else: # RGB
             image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
             
    # Resize
    image_resized = cv2.resize(image_np, IMG_SIZE)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image_resized)
    
    # Normalize and Reshape
    image_float32 = enhanced_image.astype("float32") / 255.0
    image_batch = np.expand_dims(image_float32, axis=-1) # (224, 224, 1)
    image_batch = np.expand_dims(image_batch, axis=0)    # (1, 224, 224, 1)
    
    return image_batch, enhanced_image

# --- App UI ---
st.set_page_config(page_title="Pneumonia Detector", layout="wide")

st.title("🫁 AttentiveLightNet: Pneumonia Detector")
st.markdown("This tool uses a deep learning model with attention mechanisms to detect Pneumonia from Chest X-Rays.")

model = load_app_model()

if model:
    uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Preprocess
        input_tensor, display_img = enhance_and_prepare(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(display_img, caption="Processed Input (CLAHE)", width=300)
            
        with col2:
            with st.spinner("Analyzing..."):
                prediction = model.predict(input_tensor)[0]
                class_idx = np.argmax(prediction)
                confidence = prediction[class_idx]
                result_label = CLASS_NAMES[class_idx]
                
            st.subheader("Prediction")
            if result_label == "Pneumonia":
                st.error(f"**{result_label}**")
            else:
                st.success(f"**{result_label}**")
                
            st.write(f"Confidence: **{confidence*100:.2f}%**")
            
            # Bar Chart
            probs = {name: float(prob) for name, prob in zip(CLASS_NAMES, prediction)}
            st.bar_chart(probs)