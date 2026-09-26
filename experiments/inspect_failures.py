import cv2
import csv
import ast
import math
import os

VIDEO = r".\UAV\simulation\runway_02.mp4"
CSV_FILE = r".\UAV\ground_truth_sidelines_extraction\outputrunway_02.mp4.csv"

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

ground_truth = {}

with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)

    for row in reader:
        if len(row) < 3:
            continue

        frame_number = int(row[1])
        points = ast.literal_eval(row[2])

        ground_truth[frame_number] = points

os.makedirs("failure_cases", exist_ok=True)

target_frames = [711, 716, 718, 720, 721, 727, 736, 753, 754]

for frame_number in target_frames:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    ret, image = cap.read()

    if not ret:
        print("Could not read frame:", frame_number)
        continue

    points = ground_truth[frame_number]

    gt_left_1, gt_left_2 = points[0], points[1]
    gt_right_1, gt_right_2 = points[2], points[3]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    height, width = edges.shape

    roi_top = int(height * 0.30)

    roi = edges[roi_top:height, :]

    lines = cv2.HoughLinesP(
        roi,
        1,
        math.pi / 180,
        threshold=80,
        minLineLength=80,
        maxLineGap=20
    )

    if lines is None:
        print("No lines detected:", frame_number)
        continue

    left_candidates = []
    right_candidates = []

    for line in lines:

        x1, y1, x2, y2 = line

        y1 += roi_top
        y2 += roi_top

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0 or dy == 0:
            continue

        slope = dy / dx

        length = math.sqrt(dx * dx + dy * dy)

        if length < 100:
            continue

        if slope < -0.5:
            left_candidates.append((x1, y1, x2, y2, length))

        elif slope > 0.5:
            right_candidates.append((x1, y1, x2, y2, length))

    if not left_candidates or not right_candidates:
        print("Could not detect both boundaries:", frame_number)
        continue

    left = max(left_candidates, key=lambda line: line[4])
    right = max(right_candidates, key=lambda line: line[4])

    cv2.line(image, gt_left_1, gt_left_2, (0, 0, 255), 3)
    cv2.line(image, gt_right_1, gt_right_2, (0, 0, 255), 3)

    cv2.line(
        image,
        (left[0], left[1]),
        (left[2], left[3]),
        (0, 255, 0),
        4
    )

    cv2.line(
        image,
        (right[0], right[1]),
        (right[2], right[3]),
        (0, 255, 0),
        4
    )

    output = f"failure_cases/frame_{frame_number:04d}.jpg"

    cv2.imwrite(output, image)

    print("Saved:", output)

cap.release()

print("Done.")