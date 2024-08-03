import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Title and description
st.title("Sentiment Analysis App")
st.write(" Building a Machine Learning Application with Streamlit")

# Sidebar for user input
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("", type=["png","jpg"])

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

confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

with st.spinner('Loading model...'):
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
st.success('Model loaded!')

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
st.pyplot(fig)
