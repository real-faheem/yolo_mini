import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import webbrowser

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

if st.button("About Me"):
    with st.expander("Faheem - Developer"):
        st.write(
            """
            Hi! Glad you intended to find the description about my project...
            This project is about **Detection and identification of Number Plate** under the domain of 
            **OBJECT DETECTION AND IDENTIFICATION LEVERAGING YOLOV11**.

            The need for developing a machine learning project which uses YOLOV11 to detect object-number plate
            is for the improvement of efficiency in which it can surpass the existing algorithms like CNN, RCNN etc, 
            up to 82%. Through implementing this on surveillance camera the security can be enhanced further,
            higher the efficiency - lower the setup process.
            """
        )

st.markdown("### Contacts")
phone = "9360609439"
linkedin_url = "https://www.linkedin.com/in/md-faheem-mn/"
st.markdown(f"- Phone: {phone}")
if st.button("LinkedIn Profile"):
    webbrowser.open_new_tab(linkedin_url)

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
    cap = None

    if run:
        if cap is None:
            cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to access webcam")
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_img = detect(img)
            frame_window.image(results_img)
            run = st.checkbox("Start Webcam", value=True)
        cap.release()
    else:
        if cap is not None:
            cap.release()
