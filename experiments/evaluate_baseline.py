import cv2
import csv
import ast
import math

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

total_error = 0
successful_frames = 0
failed_frames = 0

left_total_error = 0
right_total_error = 0

worst_frames = []

for frame_number in sorted(ground_truth):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    ret, image = cap.read()

    if not ret:
        failed_frames += 1
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
        failed_frames += 1
        continue

    candidates = []

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

        candidates.append((x1, y1, x2, y2, slope, length))

    left_candidates = [
        line for line in candidates
        if line[4] < -0.5
    ]

    right_candidates = [
        line for line in candidates
        if line[4] > 0.5
    ]

    best_pair = None
    best_score = float("inf")

    for left in left_candidates:

        for right in right_candidates:

            left_x_bottom = left[0] if left[1] > left[3] else left[2]
            right_x_bottom = right[0] if right[1] > right[3] else right[2]

            left_x_top = left[2] if left[1] > left[3] else left[0]
            right_x_top = right[2] if right[1] > right[3] else right[0]

            bottom_width = right_x_bottom - left_x_bottom
            top_width = right_x_top - left_x_top

            if bottom_width <= 0 or top_width <= 0:
                continue

            if bottom_width < 200 or bottom_width > 1000:
                continue

            if top_width > bottom_width:
                continue

            width_ratio = top_width / bottom_width

            score = abs(width_ratio - 0.25)

            if score < best_score:
                best_score = score
                best_pair = (left, right)

    if best_pair is None:
        failed_frames += 1
        continue

    left, right = best_pair

    def x_at_y(line, y):
        x1, y1, x2, y2, _, _ = line
        return x1 + (y - y1) * (x2 - x1) / (y2 - y1)

    predicted_left_1 = x_at_y(left, gt_left_1[1])
    predicted_left_2 = x_at_y(left, gt_left_2[1])

    predicted_right_1 = x_at_y(right, gt_right_1[1])
    predicted_right_2 = x_at_y(right, gt_right_2[1])

    left_error = (
        abs(predicted_left_1 - gt_left_1[0]) +
        abs(predicted_left_2 - gt_left_2[0])
    ) / 2

    right_error = (
        abs(predicted_right_1 - gt_right_1[0]) +
        abs(predicted_right_2 - gt_right_2[0])
    ) / 2

    frame_error = (left_error + right_error) / 2

    left_total_error += left_error
    right_total_error += right_error
    total_error += frame_error

    successful_frames += 1

    worst_frames.append((frame_error, frame_number))

cap.release()

print()
print("========== PAIR-BASED BASELINE ==========")
print("Annotated frames:", len(ground_truth))
print("Successful detections:", successful_frames)
print("Failed detections:", failed_frames)

if successful_frames > 0:

    average_left_error = left_total_error / successful_frames
    average_right_error = right_total_error / successful_frames
    average_error = total_error / successful_frames

    print()
    print(f"Average left-boundary error: {average_left_error:.2f} pixels")
    print(f"Average right-boundary error: {average_right_error:.2f} pixels")
    print(f"Overall average error: {average_error:.2f} pixels")

    worst_frames.sort(reverse=True)

    print()
    print("Worst 10 successful frames:")

    for error, frame_number in worst_frames[:10]:
        print(f"Frame {frame_number}: {error:.2f} pixels")

print()
print("Evaluation complete.")