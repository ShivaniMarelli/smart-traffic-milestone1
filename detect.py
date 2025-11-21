from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train2/weights/last.pt")  # change path

# Image input
image_path = input("Enter image path: ")

# Predict
results = model(image_path, save=True)

print("\nDetections:")
detected_any = False  # flag to check detection

for r in results:
    for box in r.boxes:
        detected_any = True
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        print(f"{cls_name} ({conf:.2f})")

# If NOTHING detected
if not detected_any:
    print("USING_MOBILE")  # your custom message

print("\nOutput saved in: runs/detect/predict/")
