import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import time
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv11 Number Plate Detection", layout="wide")

# Title and About
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Vehicle Number Plate Detection with YOLOv11")
with col2:
    with st.expander("Faheem - Developer"):
        st.markdown("""
        Hi! Glad you intended to find the description about my project...

        This project is about **"Detection and Identification of Number Plates"** under the domain of **"Object Detection and Identification leveraging YOLOv11"**.

        The need for developing a machine learning project using YOLOv11 for number plate detection is to improve the **efficiency** compared to existing algorithms like CNN, RCNN, etc., by up to **82%**.

        When implemented on surveillance cameras, this can significantly **enhance security**. Higher the efficiency, lower the setup cost.
        """)

# Contact Section
with st.sidebar:
    st.subheader("Contact")
    st.write("Phone: 9360609439")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/md-faheem-mn/)")

# Load model once
@st.cache_resource
def load_model():
    return YOLO('yolov11cus.pt')

model = load_model()

def detect(image):
    results = model(image)
    result_img = results[0].plot()
    return result_img

# Input Option
option = st.radio("Select input source:", ['Upload Image', 'Use Webcam'])

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image file (jpg/png/jpeg)", type=['jpg','jpeg','png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        results_img = detect(img_array)
        st.image(results_img, caption="Detected Number Plates")
else:
    run = st.checkbox("Start Webcam")
    if run:
        stframe = st.image([])
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not accessible")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img = detect(frame)
            stframe.image(result_img, channels="RGB")
        cap.release()
