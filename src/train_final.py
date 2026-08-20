from ultralytics import YOLO

model = YOLO("../models/yolov8n.pt")

model.train(
    data="../yolo_final/food.yaml",
    imgsz=640,
    batch=8,
    epochs=5,
    workers=2,
    device=0
)
