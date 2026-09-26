import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL = r".\runs\pose\train-4\weights\best.pt"
VIDEO = r".\UAV\simulation\runway_02.mp4"
OUTPUT = r".\outputs\runway_demo_v2.mp4"

SMOOTHING = 0.70


def extend_line_to_bottom(top, bottom, image_height):
    x1, y1 = top
    x2, y2 = bottom

    if abs(y2 - y1) < 1:
        return int(x1), int(y1), int(x2), int(y2)

    slope = (x2 - x1) / (y2 - y1)
    y_bottom = image_height - 1
    x_bottom = x1 + slope * (y_bottom - y1)

    return int(x1), int(y1), int(x_bottom), y_bottom


def main():
    model = YOLO(MODEL)

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_path = Path(OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    previous_lines = None
    frame_count = 0
    detected_count = 0

    print()
    print("========== RUNWAY VIDEO DEMO V2 ==========")
    print("Model:", MODEL)
    print("Video:", VIDEO)
    print("Output:", OUTPUT)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(
            frame,
            imgsz=640,
            device=0,
            conf=0.25,
            verbose=False
        )

        result = results[0]
        current_lines = None

        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            points = result.keypoints.xy[0].cpu().numpy()

            if len(points) == 4:
                lb, lt, rb, rt = points

                left_line = extend_line_to_bottom(
                    lt, lb, height
                )
                right_line = extend_line_to_bottom(
                    rt, rb, height
                )

                current_lines = np.array(
                    [left_line, right_line],
                    dtype=np.float32
                )

                detected_count += 1

        if current_lines is not None:
            if previous_lines is None:
                smoothed_lines = current_lines
            else:
                smoothed_lines = (
                    SMOOTHING * previous_lines
                    + (1.0 - SMOOTHING) * current_lines
                )

            previous_lines = smoothed_lines

            left_line = tuple(smoothed_lines[0].astype(int))
            right_line = tuple(smoothed_lines[1].astype(int))

            cv2.line(frame, left_line[:2], left_line[2:], (0, 255, 0), 3)
            cv2.line(frame, right_line[:2], right_line[2:], (0, 255, 0), 3)

            center_top = (
                (left_line[0] + right_line[0]) // 2,
                (left_line[1] + right_line[1]) // 2
            )
            center_bottom = (
                (left_line[2] + right_line[2]) // 2,
                (left_line[3] + right_line[3]) // 2
            )

            cv2.line(
                frame,
                center_top,
                center_bottom,
                (255, 0, 0),
                3
            )

            cv2.putText(
                frame,
                "RUNWAY DETECTED",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

        elif previous_lines is not None:
            left_line = tuple(previous_lines[0].astype(int))
            right_line = tuple(previous_lines[1].astype(int))

            cv2.line(frame, left_line[:2], left_line[2:], (0, 255, 255), 3)
            cv2.line(frame, right_line[:2], right_line[2:], (0, 255, 255), 3)

        writer.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed frames: {frame_count}")

    cap.release()
    writer.release()

    print()
    print("========== DEMO V2 COMPLETE ==========")
    print("Total frames:", frame_count)
    print("Frames with fresh 4-keypoint predictions:", detected_count)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
