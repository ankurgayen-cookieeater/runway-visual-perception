from ultralytics import YOLO
import cv2
import os
import math


MODEL = r".\runs\pose\train-2\weights\best.pt"

IMAGE_DIR = r".\yolo_runway_pose\images\test"

LABEL_DIR = r".\yolo_runway_pose\labels\test"


RANGES = [
    ("605-634", 605, 634),
    ("635-664", 635, 664),
    ("665-694", 665, 694),
    ("695-724", 695, 724),
    ("725-756", 725, 756)
]


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
            (x, y)
        )

    return keypoints


def main():

    model = YOLO(MODEL)

    print()
    print("========== YOLO ERROR BY APPROACH RANGE ==========")

    print()
    print("Model:", MODEL)

    for range_name, start_frame, end_frame in RANGES:

        keypoint_errors = [
            0,
            0,
            0,
            0
        ]

        total_error = 0

        successful = 0
        failed = 0

        for frame_number in range(
            start_frame,
            end_frame + 1
        ):

            filename = (
                f"frame_{frame_number:04d}.jpg"
            )

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

                failed += 1
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
                failed += 1
                continue

            if len(result.keypoints.xy) == 0:
                failed += 1
                continue

            predicted_points = (
                result.keypoints.xy[0]
                .cpu()
                .numpy()
            )

            if len(predicted_points) != 4:
                failed += 1
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

            total_error += frame_error / 4

            successful += 1

        print()
        print(
            "----------",
            range_name,
            "----------"
        )

        print(
            "Successful:",
            successful
        )

        print(
            "Failed:",
            failed
        )

        if successful > 0:

            averages = [
                error / successful
                for error in keypoint_errors
            ]

            overall = (
                total_error /
                successful
            )

            print(
                f"Left-bottom:  {averages[0]:.2f} pixels"
            )

            print(
                f"Left-top:     {averages[1]:.2f} pixels"
            )

            print(
                f"Right-bottom: {averages[2]:.2f} pixels"
            )

            print(
                f"Right-top:    {averages[3]:.2f} pixels"
            )

            print(
                f"Overall:      {overall:.2f} pixels"
            )

    print()
    print("========== ANALYSIS COMPLETE ==========")


if __name__ == "__main__":
    main()