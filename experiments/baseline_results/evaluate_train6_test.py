from ultralytics import YOLO

MODEL = r"..\runs\pose\train-6\weights\best.pt"
DATASET = r"..\final_yolo_pose_all_gt\runway_pose.yaml"


def main():
    print()
    print("========== TRAIN-6 TEST EVALUATION ==========")

    model = YOLO(MODEL)

    results = model.val(
        data=DATASET,
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        plots=True
    )

    print()
    print("========== TEST EVALUATION COMPLETE ==========")
    print("Model:", MODEL)
    print("Dataset:", DATASET)
    print(results)


if __name__ == "__main__":
    main()