import cv2

VIDEO = r".\outputs\runway_train7_evaluated_demo.mp4"

cap = cv2.VideoCapture(VIDEO)

frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

cap.release()

print("Frames:", frames)
print("FPS:", fps)
print("Duration:", frames / fps, "seconds")