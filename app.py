import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import altair as alt
import time

# Title and Description
st.title("Sentiment Analysis App")
st.write(" Building a Machine Learning Application with Streamlit")


st.sidebar.title("User Input")
uploaded_file = st.sidebar.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image = image.convert('L').resize((100,100))
    image_data = np.array(image).flatten()

    df = pd.DataFrame(image_data, columns=['Pixel Value'])
    df['Index'] = df.index

    X = df[['Index']].values
    y = df['Pixel Value'].values

    X_b = np.c_[np.ones((X.shape[0], 1)), X]

    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    y_pred = X_b.dot(theta_best)

    st.write("Model Coefficients:", theta_best)
    st.write("Predictions:", y_pred)

    df['Predictions'] = y_pred
    chart = alt.Chart(df).mark_line().encode(x='Index', y='Predictions')
    st.altair_chart(chart)


slider_value = st.slider("Select a value", 0, 100)
text_input = st.text_input("Enter some text")


progress_bar = st.progress(0)
for i in range(100):
    time.sleep(0.02)
    progress_bar.progress(i + 1)


data = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
chart = alt.Chart(data).mark_line().encode(x='a', y='b')
st.altair_chart(chart)
