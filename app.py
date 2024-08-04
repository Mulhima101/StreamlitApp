import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import altair as alt

# Title and Description
st.title("Food Image Classification App")
st.write("Upload an image to get the food classification")

# Sidebar for user input
st.sidebar.title("User Input")
uploaded_file = st.sidebar.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

# Class index to food name mapping
class_names = [
    "Apple Pie", "Baby Back Ribs", "Baklava", "Beef Carpaccio", "Beef Tartare", "Beet Salad", 
    "Beignets", "Bibimbap", "Bread Pudding", "Breakfast Burrito", "Bruschetta", "Caesar Salad",
    "Cannoli", "Caprese Salad", "Carrot Cake", "Ceviche", "Cheese Plate", "Cheesecake",
    "Chicken Curry", "Chicken Quesadilla", "Chicken Wings", "Chocolate Cake", "Chocolate Mousse",
    "Churros", "Clam Chowder", "Club Sandwich", "Crab Cakes", "Creme Brulee", "Croque Madame",
    "Cup Cakes", "Deviled Eggs", "Donuts", "Dumplings", "Edamame", "Eggs Benedict", "Escargots",
    "Falafel", "Filet Mignon", "Fish And Chips", "Foie Gras", "French Fries", "French Onion Soup",
    "French Toast", "Fried Calamari", "Fried Rice", "Frozen Yogurt", "Garlic Bread", "Gnocchi",
    "Greek Salad", "Grilled Cheese Sandwich", "Grilled Salmon", "Guacamole", "Gyoza", "Hamburger",
    "Hot And Sour Soup", "Hot Dog", "Huevos Rancheros", "Hummus", "Ice Cream", "Lasagna", "Lobster Bisque",
    "Lobster Roll Sandwich", "Macaroni And Cheese", "Macarons", "Miso Soup", "Mussels", "Nachos", 
    "Omelette", "Onion Rings", "Oysters", "Pad Thai", "Paella", "Pancakes", "Panna Cotta", "Peking Duck",
    "Pho", "Pizza", "Pork Chop", "Poutine", "Prime Rib", "Pulled Pork Sandwich", "Ramen", "Ravioli",
    "Red Velvet Cake", "Risotto", "Samosa", "Sashimi", "Scallops", "Seaweed Salad", "Shrimp And Grits",
    "Spaghetti Bolognese", "Spaghetti Carbonara", "Spring Rolls", "Steak", "Strawberry Shortcake",
    "Sushi", "Tacos", "Takoyaki", "Tiramisu", "Tuna Tartare", "Waffles"
]

# Function to perform convolution
def conv2d(image, kernel, bias):
    output = np.zeros((image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1, kernel.shape[3]))
    for k in range(kernel.shape[3]):
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j, k] = np.sum(image[i:i + kernel.shape[0], j:j + kernel.shape[1], :] * kernel[:, :, :, k]) + bias[k]
    return output

# Function to perform max pooling
def max_pool(image, pool_size):
    output = np.zeros((image.shape[0] // pool_size[0], image.shape[1] // pool_size[1], image.shape[2]))
    for c in range(image.shape[2]):
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j, c] = np.max(image[i * pool_size[0]:(i + 1) * pool_size[0], j * pool_size[1]:(j + 1) * pool_size[1], c])
    return output

# Function to apply ReLU activation
def relu(x):
    return np.maximum(0, x)

# Function to apply softmax activation
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# Function to flatten the image
def flatten(x):
    return x.flatten()

# Display uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Convert image to RGB and resize
    image = image.convert('RGB').resize((224, 224))
    image_data = np.array(image) / 255.0
    image_data = image_data.reshape(1, 224, 224, 3)

    # Load pre-trained model weights
    def load_model():
        return np.load('model_weights_food101.npy', allow_pickle=True).item()

    model_weights = load_model()

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Perform the forward pass with progress updates
    def predict(image_data, model_weights):
        # First convolutional layer
        status_text.text("Processing: First Convolutional Layer...")
        conv1 = conv2d(image_data[0], model_weights['conv1_kernel'], model_weights['conv1_bias'])
        conv1 = relu(conv1)
        pool1 = max_pool(conv1, (2, 2))
        progress_bar.progress(25)

        # Second convolutional layer
        status_text.text("Processing: Second Convolutional Layer...")
        conv2 = conv2d(pool1, model_weights['conv2_kernel'], model_weights['conv2_bias'])
        conv2 = relu(conv2)
        pool2 = max_pool(conv2, (2, 2))
        progress_bar.progress(50)

        # Flatten
        status_text.text("Processing: Flattening Layer...")
        flat = flatten(pool2)
        progress_bar.progress(75)

        # Fully connected layer
        status_text.text("Processing: Fully Connected Layer...")
        dense1 = np.dot(flat, model_weights['dense1_kernel']) + model_weights['dense1_bias']
        dense1 = relu(dense1)

        # Output layer
        dense2 = np.dot(dense1, model_weights['dense2_kernel']) + model_weights['dense2_bias']
        output = softmax(dense2)
        progress_bar.progress(100)

        return output

    predictions = predict(image_data, model_weights)
    predicted_class = np.argmax(predictions)
    predicted_food = class_names[predicted_class]

    st.write(f"Predicted Class Index: {predicted_class}")
    st.write(f"Predicted Food: {predicted_food}")

    progress_bar.empty()
    status_text.empty()

slider_value = st.slider("Select a value", 0, 100)
text_input = st.text_input("Enter some text")

# Progress and status updates with delay
progress_bar = st.progress(0)
for i in range(100):
    time.sleep(0.02)
    progress_bar.progress(i + 1)

data = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
chart = alt.Chart(data).mark_line().encode(x='a', y='b')
st.altair_chart(chart)
