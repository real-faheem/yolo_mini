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
        # Try multiple device indices to open webcam
        cap = None
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                break
            else:
                cap.release()
                cap = None
        if cap is None:
            st.write("Failed to access webcam on any index.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to read frame from webcam")
                    break
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_img = detect(img)
                frame_window.image(results_img)
                # Allow stopping webcam by unchecking checkbox
                run = st.checkbox("Start Webcam", value=True)
                time.sleep(0.03)  # small delay to reduce CPU load
            cap.release()
    else:
        st.write("Webcam stopped.")
