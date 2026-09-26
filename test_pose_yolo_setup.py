from ultralytics import YOLO


DATASET = r".\yolo_runway_pose\runway_pose.yaml"
MODEL = "yolo26n-pose.pt"


model = YOLO(MODEL)

print()
print("========== YOLO POSE SETUP TEST ==========")
print("Model:", MODEL)
print("Model loaded successfully.")
print("Dataset:", DATASET)
print("Model task:", model.task)
print("Model keypoint shape:", model.model.yaml.get("kpt_shape"))
print("Class names:", model.names)

print()
print("Setup test complete.")