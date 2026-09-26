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
    raise RuntimeError("Could not find both runway-side candidates.")

left = max(left_candidates, key=lambda line: line[4])
right = max(right_candidates, key=lambda line: line[4])

x1, y1, x2, y2, _ = left
cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

x1, y1, x2, y2, _ = right
cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 4)

cv2.imwrite("geometry_filter_frame_0001.jpg", image)

print("Total Hough lines:", len(lines))
print("Left candidates:", len(left_candidates))
print("Right candidates:", len(right_candidates))
print("Saved: geometry_filter_frame_0001.jpg")