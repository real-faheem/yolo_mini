from ultralytics import YOLO

model = YOLO("yolov11cus.pt")

model.predict(source = "0", show=True, save=True, conf=0.7)