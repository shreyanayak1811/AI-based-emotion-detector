import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import streamlit as st

# Load the pre-trained emotion detection model
model_path = 'emotion_model.h5'  # Update with the path to your .h5 file
model = load_model(model_path)

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize Streamlit app
st.title("Emotion Detection")
st.write("Upload an image to detect the emotion.")

# Image upload widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image using PIL
    image = Image.open(uploaded_file)
    image = image.convert("RGB")  # Ensure image is in RGB format

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process the image for emotion detection
    gray_frame = image.convert("L")  # Convert to grayscale
    face = gray_frame.resize((48, 48))  # Resize to the input size of the model
    face = np.array(face) / 255.0  # Normalize the pixel values
    face = np.expand_dims(face, axis=-1)  # Add channel dimension (48, 48, 1)
    face = np.expand_dims(face, axis=0)  # Add batch dimension (1, 48, 48, 1)

    # Predict the emotion
    predictions = model.predict(face)
    emotion_index = np.argmax(predictions[0])
    emotion = emotion_labels[emotion_index]

    # Show the predicted emotion in the Streamlit app
    st.write(f"Predicted Emotion: {emotion}")
