from ultralytics import YOLO
import os
import numpy as np

MODEL = r".\runs\pose\train-3\weights\best.pt"
IMAGE_DIR = r".\yolo_runway_pose\images\test"
LABEL_DIR = r".\yolo_runway_pose\labels\test"

IMG_WIDTH = 1280
IMG_HEIGHT = 720

RANGES = [
    (605, 634),
    (635, 664),
    (665, 694),
    (695, 724),
    (725, 756),
]


def read_ground_truth(label_path):
    with open(label_path, "r") as f:
        line = f.readline().strip()

    values = list(map(float, line.split()))

    # class + bbox + 4 keypoints
    keypoints = []

    for i in range(4):
        x = values[5 + i * 2]
        y = values[6 + i * 2]

        keypoints.append([
            x * IMG_WIDTH,
            y * IMG_HEIGHT
        ])

    return np.array(keypoints)


def main():

    print()
    print("========== YOLO 960px LOW-CONFIDENCE ERROR ==========")
    print("Model:", MODEL)
    print("Confidence threshold: 0.001")

    model = YOLO(MODEL)

    for start_frame, end_frame in RANGES:

        errors = []
        successful = 0
        failed = 0

        for frame_number in range(start_frame, end_frame + 1):

            image_path = os.path.join(
                IMAGE_DIR,
                f"frame_{frame_number:04d}.jpg"
            )

            label_path = os.path.join(
                LABEL_DIR,
                f"frame_{frame_number:04d}.txt"
            )

            if not os.path.exists(image_path):
                failed += 1
                continue

            if not os.path.exists(label_path):
                failed += 1
                continue

            gt = read_ground_truth(label_path)

            results = model.predict(
                source=image_path,
                imgsz=960,
                conf=0.001,
                device=0,
                verbose=False
            )

            result = results[0]

            if result.keypoints is None:
                failed += 1
                continue

            if len(result.keypoints.xy) == 0:
                failed += 1
                continue

            # Use the highest-confidence detection
            confidences = result.boxes.conf.cpu().numpy()
            best_index = int(np.argmax(confidences))

            pred = result.keypoints.xy[
                best_index
            ].cpu().numpy()

            if pred.shape[0] < 4:
                failed += 1
                continue

            frame_errors = np.linalg.norm(
                pred[:4] - gt[:4],
                axis=1
            )

            errors.append(frame_errors)
            successful += 1

        if successful == 0:
            print(
                f"{start_frame}-{end_frame}: "
                f"NO SUCCESSFUL PREDICTIONS"
            )
            continue

        errors = np.array(errors)

        mean_errors = np.mean(errors, axis=0)
        overall = np.mean(errors)

        print(
            f"{start_frame}-{end_frame}: "
            f"{successful} success, "
            f"{failed} fail, "
            f"LB {mean_errors[0]:.2f}, "
            f"LT {mean_errors[1]:.2f}, "
            f"RB {mean_errors[2]:.2f}, "
            f"RT {mean_errors[3]:.2f}, "
            f"overall {overall:.2f}"
        )

    print()
    print("========== EVALUATION COMPLETE ==========")


if __name__ == "__main__":
    main()