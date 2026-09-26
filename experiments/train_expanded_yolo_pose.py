from ultralytics import YOLO

MODEL = "yolo26n-pose.pt"
DATASET = r".\final_yolo_pose_all_gt\runway_pose.yaml"

def main():
    model = YOLO(MODEL)

    results = model.train(
        data=DATASET,
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0
    )

    print()
    print("========== EXPANDED TRAINING COMPLETE ==========")
    print("Model:", MODEL)
    print("Dataset:", DATASET)
    print("Training results:", results)

if __name__ == "__main__":
    main()