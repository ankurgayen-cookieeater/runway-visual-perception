import cv2
import os

VIDEO = r".\simulation\runway_02.mp4"

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

os.makedirs("inspection_frames", exist_ok=True)

for frame_number in frame_numbers:

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame:", frame_number)
        continue

    filename = f"inspection_frames/frame_{frame_number:04d}.jpg"

    cv2.imwrite(filename, frame)

    print("Saved:", filename)

cap.release()

print("Done.")