from ultralytics import YOLO
import os

MODEL = r".\runs\pose\train-3\weights\best.pt"
IMAGE_DIR = r".\yolo_runway_pose\images\test"

FRAMES = [680, 690, 700, 710, 720, 730, 740, 750]


def main():

    model = YOLO(MODEL)

    print()
    print("========== 960px LOW-CONFIDENCE DIAGNOSTIC ==========")

    for frame_number in FRAMES:

        image_path = os.path.join(
            IMAGE_DIR,
            f"frame_{frame_number:04d}.jpg"
        )

        results = model.predict(
            source=image_path,
            imgsz=960,
            conf=0.001,
            device=0,
            verbose=False
        )

        result = results[0]

        num_detections = len(result.boxes)

        print()
        print(f"Frame {frame_number}")
        print("Detections:", num_detections)

        if num_detections == 0:
            print("NO DETECTIONS")
            continue

        confidences = result.boxes.conf.cpu().numpy()

        for i, conf in enumerate(confidences):
            print(f"  Detection {i}: confidence = {conf:.6f}")

            if result.keypoints is not None:
                points = result.keypoints.xy[i].cpu().numpy()

                print("  Keypoints:")
                for j, (x, y) in enumerate(points):
                    print(
                        f"    {j}: ({x:.1f}, {y:.1f})"
                    )

    print()
    print("========== DIAGNOSTIC COMPLETE ==========")


if __name__ == "__main__":
    main()