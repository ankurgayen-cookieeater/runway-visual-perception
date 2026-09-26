from ultralytics import YOLO
import cv2
import os
import math


MODEL = r".\runs\pose\train-2\weights\best.pt"

IMAGE_DIR = r".\yolo_runway_pose\images\test"

LABEL_DIR = r".\yolo_runway_pose\labels\test"

OUTPUT_DIR = r".\yolo_worst_frames"


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

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    model = YOLO(MODEL)

    image_files = sorted(
        filename
        for filename in os.listdir(IMAGE_DIR)
        if filename.endswith(".jpg")
    )

    frame_results = []

    print()
    print("========== WORST FRAME ANALYSIS ==========")

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
            conf=0.001,
            verbose=False
        )

        result = results[0]

        if result.keypoints is None:
            continue

        if len(result.keypoints.xy) == 0:
            continue

        predicted_points = (
            result.keypoints.xy[0]
            .cpu()
            .numpy()
        )

        if len(predicted_points) != 4:
            continue

        errors = []

        for i in range(4):

            predicted = (
                float(predicted_points[i][0]),
                float(predicted_points[i][1])
            )

            errors.append(
                point_distance(
                    predicted,
                    gt_points[i]
                )
            )

        average_error = sum(errors) / 4

        confidence = float(
            result.boxes.conf[0]
            .cpu()
            .item()
        )

        frame_results.append(
            (
                average_error,
                filename,
                confidence,
                errors,
                gt_points,
                predicted_points
            )
        )

    frame_results.sort(
        reverse=True,
        key=lambda item: item[0]
    )

    print()
    print("========== TOP 10 WORST FRAMES ==========")

    for rank, item in enumerate(
        frame_results[:10],
        start=1
    ):

        (
            average_error,
            filename,
            confidence,
            errors,
            gt_points,
            predicted_points
        ) = item

        print()
        print(
            f"Rank {rank}: {filename}"
        )

        print(
            f"Confidence: {confidence:.6f}"
        )

        print(
            f"Average error: {average_error:.2f} pixels"
        )

        print(
            "Errors:"
        )

        print(
            f"  Left-bottom:  {errors[0]:.2f}"
        )

        print(
            f"  Left-top:     {errors[1]:.2f}"
        )

        print(
            f"  Right-bottom: {errors[2]:.2f}"
        )

        print(
            f"  Right-top:    {errors[3]:.2f}"
        )

        print(
            "Ground truth:",
            gt_points
        )

        print(
            "Prediction:",
            predicted_points
        )

        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        image = cv2.imread(image_path)

        for point in gt_points:

            cv2.circle(
                image,
                (
                    int(point[0]),
                    int(point[1])
                ),
                7,
                (0, 0, 255),
                -1
            )

        for point in predicted_points:

            cv2.circle(
                image,
                (
                    int(point[0]),
                    int(point[1])
                ),
                7,
                (0, 255, 0),
                -1
            )

        cv2.line(
            image,
            (
                int(gt_points[0][0]),
                int(gt_points[0][1])
            ),
            (
                int(gt_points[1][0]),
                int(gt_points[1][1])
            ),
            (0, 0, 255),
            3
        )

        cv2.line(
            image,
            (
                int(gt_points[2][0]),
                int(gt_points[2][1])
            ),
            (
                int(gt_points[3][0]),
                int(gt_points[3][1])
            ),
            (0, 0, 255),
            3
        )

        cv2.line(
            image,
            (
                int(predicted_points[0][0]),
                int(predicted_points[0][1])
            ),
            (
                int(predicted_points[1][0]),
                int(predicted_points[1][1])
            ),
            (0, 255, 0),
            3
        )

        cv2.line(
            image,
            (
                int(predicted_points[2][0]),
                int(predicted_points[2][1])
            ),
            (
                int(predicted_points[3][0]),
                int(predicted_points[3][1])
            ),
            (0, 255, 0),
            3
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        cv2.imwrite(
            output_path,
            image
        )

    print()
    print(
        "Worst-frame visualizations saved to:",
        OUTPUT_DIR
    )

    print()
    print("Analysis complete.")


if __name__ == "__main__":
    main()