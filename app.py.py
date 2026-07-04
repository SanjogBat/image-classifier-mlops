import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Intel Image Classifier API",
    page_icon="🌍",
    layout="centered"
)

@st.cache_resource
def load_model():
    """
    Loads the trained CNN model. 
    @st.cache_resource ensures the model is only loaded into RAM once, 
    preventing massive slowdowns on every user interaction.
    """
    return tf.keras.models.load_model('models/cnn_classifier.h5')

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model. Please ensure models/cnn_classifier.h5 exists. Error: {e}")
    st.stop()

CLASS_NAMES = ['Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street']

st.title("🌍 Scenery Classification AI")
st.markdown("""
This application uses a custom-trained **Convolutional Neural Network (CNN)** to classify landscape images. 
Upload an image, and the model will predict its category using vectorized inference.
""")

uploaded_file = st.file_uploader("Upload a landscape image (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image and strip Alpha channel to prevent tensor shape mismatches
    image = Image.open(uploaded_file).convert('RGB')
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    with st.spinner('Running AI Inference...'):
        # 1. Resize to match the 150x150 input shape defined in train_model.py
        img_resized = image.resize((150, 150))
        
        # 2. Convert to NumPy array and Normalize
        img_array = np.array(img_resized) / 255.0
        
        # 3. Vectorization: Add batch dimension (1, 150, 150, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        
        # 4. Predict
        predictions = model.predict(img_batch)
        
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = np.max(predictions[0]) * 100
    
    # Display dynamic UI based on confidence threshold
    if confidence > 75:
        st.success(f"### Prediction: **{predicted_class}**")
        st.write(f"**Confidence:** {confidence:.2f}%")
    else:
        st.warning(f"### Prediction: **{predicted_class}**")
        st.write(f"**Confidence:** {confidence:.2f}% (Low confidence - image may be ambiguous)")
        
    st.markdown("---")
    st.write("### Probability Distribution")
    st.bar_chart(dict(zip(CLASS_NAMES, predictions[0])))