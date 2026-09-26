import cv2
import math
import os

VIDEO = r".\UAV\simulation\runway_02.mp4"

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_numbers = [
    1,
    total_frames // 4,
    total_frames // 2,
    (3 * total_frames) // 4,
    total_frames - 1
]

os.makedirs("baseline_results", exist_ok=True)

for frame_number in frame_numbers:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    ret, image = cap.read()

    if not ret:
        print("Could not read frame:", frame_number)
        continue

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
        print("No lines detected in frame:", frame_number)
        continue

    left_candidates = []
    right_candidates = []

    for line in lines:

        x1, y1, x2, y2 = line

        y1 += roi_top
        y2 += roi_top

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
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
        print("Could not find both boundaries in frame:", frame_number)
        continue

    left = max(left_candidates, key=lambda line: line[4])
    right = max(right_candidates, key=lambda line: line[4])

    x1, y1, x2, y2, _ = left
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    x1, y1, x2, y2, _ = right
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    output = f"baseline_results/frame_{frame_number:04d}.jpg"

    cv2.imwrite(output, image)

    print(
        "Frame:",
        frame_number,
        "| Hough lines:",
        len(lines),
        "| Left candidates:",
        len(left_candidates),
        "| Right candidates:",
        len(right_candidates)
    )

cap.release()

print("Done.")