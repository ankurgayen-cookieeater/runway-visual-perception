from ultralytics import YOLO
import cv2
import os
import math


MODEL = r".\runs\pose\train-2\weights\best.pt"

IMAGE_DIR = r".\yolo_runway_pose\images\test"

LABEL_DIR = r".\yolo_runway_pose\labels\test"


def point_distance(point1, point2):

    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]

    return math.sqrt(
        dx * dx + dy * dy
    )


def read_label(label_path, width, height):

    with open(label_path, "r") as file:

        values = list(
            map(
                float,
                file.readline().split()
            )
        )

    keypoints = []

    for i in range(5, 13, 2):

        x = values[i] * width
        y = values[i + 1] * height

        keypoints.append(
            (
                x,
                y
            )
        )

    return keypoints


def main():

    model = YOLO(MODEL)

    image_files = sorted(
        filename
        for filename in os.listdir(IMAGE_DIR)
        if filename.endswith(".jpg")
    )

    print()
    print("========== YOLO TEST IMAGE EVALUATION ==========")
    print("Model:", MODEL)
    print("Test images:", len(image_files))

    keypoint_errors = [
        0,
        0,
        0,
        0
    ]

    total_error = 0

    successful_frames = 0
    failed_frames = 0

    for filename in image_files:

        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        label_path = os.path.join(
            LABEL_DIR,
            filename.replace(
                ".jpg",
                ".txt"
            )
        )

        image = cv2.imread(image_path)

        if image is None:

            print(
                "Could not read:",
                filename
            )

            failed_frames += 1
            continue

        height, width = image.shape[:2]

        gt_points = read_label(
            label_path,
            width,
            height
        )

        results = model(
            image,
            device=0,
            verbose=False
        )

        result = results[0]

        if result.keypoints is None:

            failed_frames += 1
            continue

        if len(result.keypoints.xy) == 0:

            failed_frames += 1
            continue

        predicted_points = (
            result.keypoints.xy[0]
            .cpu()
            .numpy()
        )

        if len(predicted_points) != 4:

            failed_frames += 1
            continue

        frame_error = 0

        for i in range(4):

            predicted = (
                float(predicted_points[i][0]),
                float(predicted_points[i][1])
            )

            error = point_distance(
                predicted,
                gt_points[i]
            )

            keypoint_errors[i] += error

            frame_error += error

        frame_error /= 4

        total_error += frame_error

        successful_frames += 1

    print()
    print("========== RESULTS ==========")

    print(
        "Successful detections:",
        successful_frames
    )

    print(
        "Failed detections:",
        failed_frames
    )

    if successful_frames > 0:

        average_errors = [
            error / successful_frames
            for error in keypoint_errors
        ]

        overall_error = (
            total_error /
            successful_frames
        )

        print()
        print(
            f"Left-bottom error:  "
            f"{average_errors[0]:.2f} pixels"
        )

        print(
            f"Left-top error:     "
            f"{average_errors[1]:.2f} pixels"
        )

        print(
            f"Right-bottom error: "
            f"{average_errors[2]:.2f} pixels"
        )

        print(
            f"Right-top error:    "
            f"{average_errors[3]:.2f} pixels"
        )

        print()
        print(
            f"Overall keypoint error: "
            f"{overall_error:.2f} pixels"
        )

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()