from ultralytics import YOLO

model = YOLO("yolo11m.pt")

model.train(data = "data_custom.yaml", imgsz = 640, batch = 8, epochs = 20, workers = 0, device = 0)