import cv2

IMAGE = r".\inspection_frames\frame_0001.jpg"

image = cv2.imread(IMAGE)

if image is None:
    raise RuntimeError("Could not open image.")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

cv2.imwrite("canny_frame_0001.jpg", edges)

print("Saved: canny_frame_0001.jpg")