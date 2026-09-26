import cv2
import csv
import ast
import os


VIDEO = r".\UAV\simulation\runway_02.mp4"
CSV_FILE = r".\UAV\ground_truth_sidelines_extraction\outputrunway_02.mp4.csv"

OUTPUT_ROOT = r".\yolo_runway_pose"


TRAIN_START = 1
TRAIN_END = 453

VAL_START = 454
VAL_END = 604

TEST_START = 605
TEST_END = 756


def create_directories():

    directories = [
        os.path.join(OUTPUT_ROOT, "images", "train"),
        os.path.join(OUTPUT_ROOT, "images", "val"),
        os.path.join(OUTPUT_ROOT, "images", "test"),
        os.path.join(OUTPUT_ROOT, "labels", "train"),
        os.path.join(OUTPUT_ROOT, "labels", "val"),
        os.path.join(OUTPUT_ROOT, "labels", "test")
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_split(frame_number):

    if TRAIN_START <= frame_number <= TRAIN_END:
        return "train"

    if VAL_START <= frame_number <= VAL_END:
        return "val"

    if TEST_START <= frame_number <= TEST_END:
        return "test"

    return None


def normalize(value, maximum):

    return value / maximum


def create_yolo_label(points, width, height):

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x_min = min(xs)
    x_max = max(xs)

    y_min = min(ys)
    y_max = max(ys)

    bbox_x = (x_min + x_max) / 2
    bbox_y = (y_min + y_max) / 2

    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    bbox_x = normalize(bbox_x, width)
    bbox_y = normalize(bbox_y, height)
    bbox_width = normalize(bbox_width, width)
    bbox_height = normalize(bbox_height, height)

    keypoints = []

    for x, y in points:

        keypoints.append(normalize(x, width))
        keypoints.append(normalize(y, height))

    values = [
        0,
        bbox_x,
        bbox_y,
        bbox_width,
        bbox_height
    ]

    values.extend(keypoints)

    return " ".join(f"{value:.6f}" for value in values)


def main():

    create_directories()

    ground_truth = {}

    with open(CSV_FILE, "r") as file:

        reader = csv.reader(file)

        for row in reader:

            if len(row) < 3:
                continue

            frame_number = int(row[1])

            points = ast.literal_eval(row[2])

            ground_truth[frame_number] = points

    print("Annotated frames:", len(ground_truth))

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video width:", width)
    print("Video height:", height)
    print("Total video frames:", total_frames)

    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    for frame_number in sorted(ground_truth):

        split = get_split(frame_number)

        if split is None:
            continue

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1
        )

        ret, frame = cap.read()

        if not ret:
            print("Could not read frame:", frame_number)
            continue

        points = ground_truth[frame_number]

        label = create_yolo_label(
            points,
            width,
            height
        )

        image_filename = (
            f"frame_{frame_number:04d}.jpg"
        )

        label_filename = (
            f"frame_{frame_number:04d}.txt"
        )

        image_path = os.path.join(
            OUTPUT_ROOT,
            "images",
            split,
            image_filename
        )

        label_path = os.path.join(
            OUTPUT_ROOT,
            "labels",
            split,
            label_filename
        )

        cv2.imwrite(
            image_path,
            frame
        )

        with open(label_path, "w") as file:

            file.write(label + "\n")

        split_counts[split] += 1

    cap.release()

    print()
    print("========== DATASET PREPARATION ==========")
    print("Train images:", split_counts["train"])
    print("Validation images:", split_counts["val"])
    print("Test images:", split_counts["test"])
    print("Total:", sum(split_counts.values()))
    print()
    print("Dataset saved to:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()