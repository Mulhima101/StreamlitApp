import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Title and description
st.title("Sentiment Analysis App")
st.write(" Building a Machine Learning Application with Streamlit")

# Load dataset and train model
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"])

# Display the uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    image = image.convert('L').resize((8, 8))
    img_array = np.array(image).reshape(1, -1)
    st.write("")
    st.write("Classifying...")

    # Normalize the image array
    img_array = img_array / 16.0

    # Make predictions
    prediction = model.predict(img_array)
    st.write(f"Predicted Digit: {prediction[0]}")

# Display model performance
st.write("Model Performance:")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy: {accuracy:.2f}")

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm).plot(ax=ax)
st.pyplot(fig)
