import cv2
import math

IMAGE = r".\inspection_frames\frame_0001.jpg"

image = cv2.imread(IMAGE)

if image is None:
    raise RuntimeError("Could not open image.")

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
    raise RuntimeError("No lines detected.")

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
    raise RuntimeError("Could not find a suitable runway pair.")

left, right = best_pair

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

cv2.imwrite("pair_geometry_frame_0001.jpg", image)

print("Left candidates:", len(left_candidates))
print("Right candidates:", len(right_candidates))
print("Best pair score:", best_score)
print("Saved: pair_geometry_frame_0001.jpg")