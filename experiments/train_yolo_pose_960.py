from ultralytics import YOLO


MODEL = "yolo26n-pose.pt"
DATASET = r".\yolo_runway_pose\runway_pose.yaml"


def main():

    model = YOLO(MODEL)

    results = model.train(
        data=DATASET,
        epochs=50,
        imgsz=960,
        batch=4,
        device=0,
        patience=15,
        workers=0
    )

    print()
    print("========== 960px TRAINING COMPLETE ==========")
    print("Model:", MODEL)
    print("Dataset:", DATASET)
    print("Image size:", 960)
    print("Training results:", results)


if __name__ == "__main__":
    main()