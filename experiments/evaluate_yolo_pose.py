from ultralytics import YOLO
import cv2
import csv
import ast
import math
import os


MODEL = r".\runs\pose\train\weights\best.pt"

VIDEO = r".\UAV\simulation\runway_02.mp4"

CSV_FILE = r".\UAV\ground_truth_sidelines_extraction\outputrunway_02.mp4.csv"

OUTPUT_DIR = r".\yolo_pose_test_results"


TEST_START = 605
TEST_END = 756


def point_distance(point1, point2):

    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]

    return math.sqrt(dx * dx + dy * dy)


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(MODEL)

    ground_truth = {}

    with open(CSV_FILE, "r") as file:

        reader = csv.reader(file)

        for row in reader:

            if len(row) < 3:
                continue

            frame_number = int(row[1])

            points = ast.literal_eval(row[2])

            ground_truth[frame_number] = points

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    total_error = 0

    keypoint_errors = [0, 0, 0, 0]

    successful_frames = 0

    failed_frames = 0

    for frame_number in range(TEST_START, TEST_END + 1):

        if frame_number not in ground_truth:
            failed_frames += 1
            continue

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1
        )

        ret, frame = cap.read()

        if not ret:
            print("Could not read frame:", frame_number)
            failed_frames += 1
            continue

        results = model(
            frame,
            device=0,
            verbose=False
        )

        result = results[0]

        if result.keypoints is None:
            print("No keypoints detected in frame:", frame_number)
            failed_frames += 1
            continue

        if len(result.keypoints.xy) == 0:
            print("No runway detected in frame:", frame_number)
            failed_frames += 1
            continue

        predicted_points = result.keypoints.xy[0].cpu().numpy()

        if len(predicted_points) != 4:
            print(
                "Incorrect number of keypoints in frame:",
                frame_number,
                "| Detected:",
                len(predicted_points)
            )
            failed_frames += 1
            continue

        predicted_points = [
            (
                float(point[0]),
                float(point[1])
            )
            for point in predicted_points
        ]

        gt_points = ground_truth[frame_number]

        frame_error = 0

        for i in range(4):

            error = point_distance(
                predicted_points[i],
                gt_points[i]
            )

            keypoint_errors[i] += error

            frame_error += error

        frame_error /= 4

        total_error += frame_error

        successful_frames += 1

        if frame_number in [605, 630, 680, 730, 756]:

            output = frame.copy()

            gt_colors = [
                (0, 0, 255),
                (0, 0, 255),
                (0, 0, 255),
                (0, 0, 255)
            ]

            for point in gt_points:

                cv2.circle(
                    output,
                    point,
                    7,
                    (0, 0, 255),
                    -1
                )

            for point in predicted_points:

                cv2.circle(
                    output,
                    (
                        int(point[0]),
                        int(point[1])
                    ),
                    7,
                    (0, 255, 0),
                    -1
                )

            cv2.line(
                output,
                gt_points[0],
                gt_points[1],
                (0, 0, 255),
                3
            )

            cv2.line(
                output,
                gt_points[2],
                gt_points[3],
                (0, 0, 255),
                3
            )

            cv2.line(
                output,
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
                output,
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
                f"frame_{frame_number:04d}.jpg"
            )

            cv2.imwrite(
                output_path,
                output
            )

    cap.release()

    print()
    print("========== YOLO POSE TEST EVALUATION ==========")
    print("Test frames:", TEST_END - TEST_START + 1)
    print("Successful detections:", successful_frames)
    print("Failed detections:", failed_frames)

    if successful_frames > 0:

        average_errors = [
            error / successful_frames
            for error in keypoint_errors
        ]

        overall_error = (
            total_error / successful_frames
        )

        print()
        print(
            f"Left-bottom error:  {average_errors[0]:.2f} pixels"
        )

        print(
            f"Left-top error:     {average_errors[1]:.2f} pixels"
        )

        print(
            f"Right-bottom error: {average_errors[2]:.2f} pixels"
        )

        print(
            f"Right-top error:    {average_errors[3]:.2f} pixels"
        )

        print()
        print(
            f"Overall keypoint error: {overall_error:.2f} pixels"
        )

    print()
    print(
        "Visualizations saved to:",
        OUTPUT_DIR
    )

    print()
    print("Evaluation complete.")


if __name__ == "__main__":
    main()