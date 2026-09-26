import cv2

IMAGE = r".\inspection_frames\frame_0001.jpg"

image = cv2.imread(IMAGE)

if image is None:
    raise RuntimeError("Could not open image.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

lines = cv2.HoughLinesP(
    edges,
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

    cv2.line(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

cv2.imwrite("hough_frame_0001.jpg", image)

print("Detected lines:", len(lines))
print("Saved: hough_frame_0001.jpg")