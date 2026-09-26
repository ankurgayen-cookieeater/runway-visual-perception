from ultralytics import YOLO
import os
import numpy as np
import csv

MODEL_640 = r".\runs\pose\train-2\weights\best.pt"
MODEL_960 = r".\runs\pose\train-3\weights\best.pt"

IMAGE_DIR = r".\yolo_runway_pose\images\test"
LABEL_DIR = r".\yolo_runway_pose\labels\test"

IMG_WIDTH = 1280
IMG_HEIGHT = 720

EVAL_Y = 715

RANGES = [
    (605, 634),
    (635, 664),
    (665, 694),
    (695, 724),
    (725, 756),
]


def line_x_at_y(p1, p2, y):
    x1, y1 = p1
    x2, y2 = p2

    if abs(y2 - y1) < 1e-6:
        return (x1 + x2) / 2

    x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)

    return x


def read_ground_truth(label_path):

    with open(label_path, "r") as f:
        values = list(map(float, f.readline().split()))

    points = []

    for i in range(4):
        x = values[5 + i * 2] * IMG_WIDTH
        y = values[6 + i * 2] * IMG_HEIGHT

        points.append([x, y])

    return np.array(points)


def get_prediction(model, image_path, imgsz):

    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=0.001,
        device=0,
        verbose=False
    )

    result = results[0]

    if result.keypoints is None:
        return None, None

    if len(result.keypoints.xy) == 0:
        return None, None

    confidences = result.boxes.conf.cpu().numpy()

    best_index = int(np.argmax(confidences))

    points = result.keypoints.xy[
        best_index
    ].cpu().numpy()

    confidence = float(confidences[best_index])

    return points, confidence


def calculate_width(points):

    left_x = line_x_at_y(
        points[0],
        points[1],
        EVAL_Y
    )

    right_x = line_x_at_y(
        points[2],
        points[3],
        EVAL_Y
    )

    width = right_x - left_x

    return left_x, right_x, width


def main():

    print()
    print("========== RUNWAY WIDTH ANALYSIS ==========")
    print("Evaluation Y:", EVAL_Y)

    model_640 = YOLO(MODEL_640)
    model_960 = YOLO(MODEL_960)

    rows = []

    for frame_number in range(605, 757):

        image_path = os.path.join(
            IMAGE_DIR,
            f"frame_{frame_number:04d}.jpg"
        )

        label_path = os.path.join(
            LABEL_DIR,
            f"frame_{frame_number:04d}.txt"
        )

        if not os.path.exists(image_path):
            continue

        if not os.path.exists(label_path):
            continue

        gt = read_ground_truth(label_path)

        gt_left, gt_right, gt_width = calculate_width(gt)

        pred640, conf640 = get_prediction(
            model_640,
            image_path,
            640
        )

        pred960, conf960 = get_prediction(
            model_960,
            image_path,
            960
        )

        if pred640 is None or pred960 is None:
            continue

        p640_left, p640_right, p640_width = calculate_width(
            pred640
        )

        p960_left, p960_right, p960_width = calculate_width(
            pred960
        )

        rows.append({
            "frame": frame_number,

            "gt_width": gt_width,

            "width_640": p640_width,
            "width_error_640": abs(p640_width - gt_width),

            "width_960": p960_width,
            "width_error_960": abs(p960_width - gt_width),

            "gt_left_x": gt_left,
            "gt_right_x": gt_right,

            "left_640_x": p640_left,
            "right_640_x": p640_right,

            "left_960_x": p960_left,
            "right_960_x": p960_right,

            "conf_640": conf640,
            "conf_960": conf960
        })

    print()
    print("Total frames:", len(rows))

    print()
    print("---------- OVERALL ----------")

    gt_widths = np.array([r["gt_width"] for r in rows])
    width640 = np.array([r["width_640"] for r in rows])
    width960 = np.array([r["width_960"] for r in rows])

    error640 = np.abs(width640 - gt_widths)
    error960 = np.abs(width960 - gt_widths)

    print(f"GT mean width:   {np.mean(gt_widths):.2f} px")
    print(f"640 mean width:  {np.mean(width640):.2f} px")
    print(f"960 mean width:  {np.mean(width960):.2f} px")

    print()
    print(f"640 mean width error: {np.mean(error640):.2f} px")
    print(f"960 mean width error: {np.mean(error960):.2f} px")

    print()
    print("---------- BY APPROACH RANGE ----------")

    for start, end in RANGES:

        subset = [
            r for r in rows
            if start <= r["frame"] <= end
        ]

        if not subset:
            continue

        gt = np.array([
            r["gt_width"] for r in subset
        ])

        w640 = np.array([
            r["width_640"] for r in subset
        ])

        w960 = np.array([
            r["width_960"] for r in subset
        ])

        e640 = np.abs(w640 - gt)
        e960 = np.abs(w960 - gt)

        print()
        print(f"{start}-{end}")
        print(f"  GT width:       {np.mean(gt):.2f} px")
        print(f"  640 width:      {np.mean(w640):.2f} px")
        print(f"  640 error:      {np.mean(e640):.2f} px")
        print(f"  960 width:      {np.mean(w960):.2f} px")
        print(f"  960 error:      {np.mean(e960):.2f} px")

    output_csv = "runway_width_analysis.csv"

    if rows:

        fieldnames = rows[0].keys()

        with open(
            output_csv,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(rows)

    print()
    print("CSV saved:", output_csv)

    print()
    print("========== WIDTH ANALYSIS COMPLETE ==========")


if __name__ == "__main__":
    main()