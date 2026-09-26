import cv2

IMAGE = r".\inspection_frames\frame_0001.jpg"

image = cv2.imread(IMAGE)

if image is None:
    raise RuntimeError("Could not open image.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

height, width = edges.shape

roi = edges[int(height * 0.30):height, :]

lines = cv2.HoughLinesP(
    roi,
    1,
    3.14159 / 180,
    threshold=80,
    minLineLength=80,
    maxLineGap=20
)

if lines is None:
    raise RuntimeError("No lines detected.")

for line in lines:
    x1, y1, x2, y2 = line

    y1 += int(height * 0.30)
    y2 += int(height * 0.30)

    cv2.line(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

cv2.imwrite("roi_hough_frame_0001.jpg", image)

print("Detected lines:", len(lines))
print("Saved: roi_hough_frame_0001.jpg")