from ultralytics import YOLO

MODEL = r".\runs\pose\train-5\weights\best.pt"
DATASET = r".\final_yolo_pose\runway_pose.yaml"


def main():
    print()
    print("========== FINAL TEST EVALUATION ==========")

    model = YOLO(MODEL)

    results = model.val(
        data=DATASET,
        split="test",
        imgsz=960,
        batch=8,
        device=0,
        workers=0,
        plots=True
    )

    print()
    print("========== FINAL TEST COMPLETE ==========")
    print("Model:", MODEL)
    print("Dataset:", DATASET)
    print(results)


if __name__ == "__main__":
    main()
