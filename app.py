import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import time

st.title("Vehicle Number Plate Detection with YOLOv11")

option = st.radio("Select input source:", ['Upload Image', 'Use Webcam'])

@st.cache_resource
def load_model():
    from ultralytics import YOLO
    model = YOLO('yolov11cus.pt')
    return model

model = load_model()

def detect(image):
    results = model(image)
    result_img = results[0].plot()
    return result_img

if option == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image file (jpg/png/jpeg)", type=['jpg','jpeg','png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        results_img = detect(img_array)
        st.image(results_img, caption="Detected Number Plates")

else:
    run = st.checkbox("Start Webcam")
    frame_window = st.image([])

    if run:
        # Instead of cv2.VideoCapture, use Streamlit's experimental camera_input widget for better compatibility
        cam_file = st.camera_input("Use your camera to capture live video")
        if cam_file is not None:
            bytes_data = cam_file.getvalue()
            np_arr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_img = detect(img)
            frame_window.image(results_img)
        else:
            st.write("Please allow camera access.")
    else:
        st.write("Webcam stopped.")
