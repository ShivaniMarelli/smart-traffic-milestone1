from ultralytics import YOLO

# Load pretrained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train the model on your dataset
model.train(
    data="datasetsyolo/data.yaml",  # path to your fixed data.yaml
    epochs=50,
    imgsz=640,
    batch=8
)
