import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.title("Sentiment Analysis App")
st.write(" Building a Machine Learning Application with Streamlit")


st.sidebar.title("User Input")
uploaded_file = st.sidebar.file_uploader("Upload a file", type=["csv", "png", "jpg", "mp4", "mp3"])


if uploaded_file is not None:
    file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    if uploaded_file.type == "image/png" or uploaded_file.type == "image/jpeg":
        st.image(uploaded_file)
    elif uploaded_file.type == "video/mp4":
        st.video(uploaded_file)
    elif uploaded_file.type == "audio/mp3":
        st.audio(uploaded_file)


slider_value = st.slider("Select a value", 0, 100)
text_input = st.text_input("Enter some text")


progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)


data = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
chart = alt.Chart(data).mark_line().encode(x='a', y='b')
st.altair_chart(chart)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
X = data[['a']]
y = data['b']
model.fit(X, y)
predictions = model.predict(X)
st.write("Model Predictions:", predictions)
