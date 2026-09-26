from ultralytics import YOLO

MODEL = "yolo26n-pose.pt"
DATASET = r".\final_yolo_pose_mixed_domain\runway_pose.yaml"


def main():
    print()
    print("========== MIXED-DOMAIN TRAINING ==========")

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
    print("========== MIXED-DOMAIN TRAINING COMPLETE ==========")
    print("Model:", MODEL)
    print("Dataset:", DATASET)
    print("Training results:", results)


if __name__ == "__main__":
    main()
