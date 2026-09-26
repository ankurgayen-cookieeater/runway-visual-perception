from ultralytics import YOLO
import cv2
import numpy as np
import os

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
        values = list(map(float, f.readline().split()))

    # class + bbox(4) + 4 keypoints(x,y)
    keypoints = []

    for i in range(4):
        x = values[5 + i * 2] * IMG_WIDTH
        y = values[6 + i * 2] * IMG_HEIGHT
        keypoints.append([x, y])

    return np.array(keypoints)


def main():
    print()
    print("========== YOLO 960px ERROR BY APPROACH RANGE ==========")

    model = YOLO(MODEL)

    print("Model:", MODEL)
    print()

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

            if not os.path.exists(image_path) or not os.path.exists(label_path):
                failed += 1
                continue

            ground_truth = read_ground_truth(label_path)

            results = model.predict(
                source=image_path,
                imgsz=960,
                device=0,
                verbose=False
            )

            if len(results) == 0:
                failed += 1
                continue

            keypoints = results[0].keypoints

            if keypoints is None or len(keypoints.xy) == 0:
                failed += 1
                continue

            predicted = keypoints.xy[0].cpu().numpy()

            if predicted.shape[0] != 4:
                failed += 1
                continue

            point_errors = np.linalg.norm(
                predicted - ground_truth,
                axis=1
            )

            errors.append(point_errors)
            successful += 1

        if successful == 0:
            print(
                f"{start_frame}-{end_frame}: "
                f"NO SUCCESSFUL PREDICTIONS"
            )
            continue

        errors = np.array(errors)

        left_bottom = errors[:, 0].mean()
        left_top = errors[:, 1].mean()
        right_bottom = errors[:, 2].mean()
        right_top = errors[:, 3].mean()
        overall = errors.mean()

        print(
            f"{start_frame}-{end_frame}: "
            f"{successful} success, {failed} fail, "
            f"LB {left_bottom:.2f}, "
            f"LT {left_top:.2f}, "
            f"RB {right_bottom:.2f}, "
            f"RT {right_top:.2f}, "
            f"overall {overall:.2f}"
        )

    print()
    print("========== EVALUATION COMPLETE ==========")


if __name__ == "__main__":
    main()