from ultralytics import YOLO

model = YOLO("../models/yolov8n.pt")   

model.train(
    data="food.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device="cpu",
    name="food_nano"
)
