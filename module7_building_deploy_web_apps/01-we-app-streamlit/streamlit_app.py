import streamlit as st

st.title("OpenCV Streamlit Demo")
st.header("header")
image = st.file_uploader("Upload an image file")
st.image(image)

st.text("You develop Computer Vision apps with frameworks and libraries such as TensorFlow, Pytorch and OpenCV.")
selected_value = st.selectbox("Select the channel", ["R (red)", "G (green)", "B (blue)"])

st.write(selected_value)

checkbox_value = st.checkbox("Apply Filter")

st.write(checkbox_value)