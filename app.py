import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np

st.set_page_config(page_title="Vehicle Number Plate Detection", layout="centered")

# Description Section
with st.sidebar:
    st.title("Faheem - Developer")
    with st.expander("About Project"):
        st.markdown("""
        Hi! Glad you intended to find the description about my project...

        **This project is about**: "Detection and identification of Number Plate"  
        **Domain**: OBJECT DETECTION AND IDENTIFICATION LEVERAGING YOLOV11.

        The need for developing a machine learning project which uses **YOLOV11** to detect object-number plate is for the improvement of efficiency.  
        It can surpass existing algorithms like **CNN, RCNN**, etc., by up to **82%**.

        Through implementing this on surveillance cameras, **security can be enhanced** further.  
        **Higher the efficiency → Lower the setup process.**
        """)
    
    st.markdown("### 📬 Contact")
    st.markdown('[📧 faheemhaker@gmail.com]')
    st.markdown('[🔗 LinkedIn](https://www.linkedin.com/in/md-faheem-mn/)', unsafe_allow_html=True)

# Model loader
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

# Input Option
st.title("Vehicle Number Plate Detection with YOLOv11")
option = st.radio("Select input source:", ['Upload Image', 'Use Webcam'])

# Upload Image Mode
if option == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image file", type=['jpg','jpeg','png'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)
        result_img = detect(img_array)
        st.image(result_img, caption="Detected Number Plates", use_container_width=True)

# Webcam Mode (Mobile + Laptop supported)
else:
    st.info("Use camera input below. Works on both mobile and laptop.")
    cam_input = st.camera_input("Take a picture")
    if cam_input:
        image = Image.open(cam_input).convert('RGB')
        img_array = np.array(image)
        result_img = detect(img_array)
        st.image(result_img, caption="Detected Number Plates", use_container_width=True)
