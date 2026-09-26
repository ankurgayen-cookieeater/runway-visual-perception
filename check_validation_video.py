import cv2

VIDEO = r".\UAV\real_videos\runway_video20230228-103640.mp4"

cap = cv2.VideoCapture(VIDEO)

print("Opened:", cap.isOpened())
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frames:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()