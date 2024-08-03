import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Title and description
st.title("Image Classification with Streamlit")
st.write("Upload an image to classify it using a pre-trained model.")

# Sidebar for user input
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image and make predictions
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    img_array = np.array(image.resize((224, 224))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    st.write("Predictions:", predictions)
