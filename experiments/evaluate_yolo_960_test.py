from ultralytics import YOLO


MODEL = r".\runs\pose\train-3\weights\best.pt"

DATASET = r".\yolo_runway_pose\runway_pose.yaml"


def main():

    print()
    print("========== YOLO 960px TEST EVALUATION ==========")

    model = YOLO(MODEL)

    print("Model:", MODEL)
    print("Dataset:", DATASET)

    results = model.val(
        data=DATASET,
        split="test",
        imgsz=960,
        batch=4,
        device=0,
        workers=0,
        plots=True
    )

    print()
    print("========== 960px TEST EVALUATION COMPLETE ==========")
    print(results)


if __name__ == "__main__":
    main()