from ultralytics import YOLO


MODEL = "yolo26n-pose.pt"
DATASET = r".\yolo_runway_pose\runway_pose.yaml"


def main():

    print()
    print("========== POSE DATASET SANITY CHECK ==========")

    model = YOLO(MODEL)

    print("Model loaded:", MODEL)

    model.train(
        data=DATASET,
        epochs=1,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=False,
        save=False
    )

    print()
    print("========== SANITY CHECK COMPLETE ==========")


if __name__ == "__main__":
    main()