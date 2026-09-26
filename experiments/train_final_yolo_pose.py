from ultralytics import YOLO

MODEL = "yolo26n-pose.pt"
DATASET = r".\final_yolo_pose\runway_pose.yaml"


def main():
    model = YOLO(MODEL)

    results = model.train(
        data=DATASET,
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        patience=15,
        workers=0
    )

    print()
    print("========== FINAL TRAINING COMPLETE ==========")
    print("Model:", MODEL)
    print("Dataset:", DATASET)
    print("Training results:", results)


if __name__ == "__main__":
    main()