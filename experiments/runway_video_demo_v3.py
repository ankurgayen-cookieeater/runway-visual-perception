import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL = r".\runs\pose\train-4\weights\best.pt"
VIDEO = r".\UAV\simulation\runway_02.mp4"
OUTPUT = r".\outputs\runway_demo_v3.mp4"

SMOOTHING = 0.65
ANGLE_TOLERANCE = 12.0
TOP_DISTANCE = 80.0


def line_x_at_y(x1, y1, x2, y2, y):
    if abs(y2 - y1) < 1:
        return None
    return x1 + (x2 - x1) * (y - y1) / (y2 - y1)


def refine_boundary(frame, top, bottom, side):
    height, width = frame.shape[:2]

    x1, y1 = top
    x2, y2 = bottom

    predicted_angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 160)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=max(80, height // 8),
        maxLineGap=35
    )

    best = None
    best_score = float("inf")

    if lines is None:
        return (x1, y1, x2, y2)

    for line in lines.reshape(-1, 4):
        a, b, c, d = map(float, line)

        angle = np.degrees(np.arctan2(d - b, c - a))

        # Treat a line and the same line with reversed endpoints identically.
        if angle < -90:
            angle += 180
        if angle > 90:
            angle -= 180

        target_angle = predicted_angle
        if target_angle < -90:
            target_angle += 180
        if target_angle > 90:
            target_angle -= 180

        angle_error = abs(angle - target_angle)
        if angle_error > 180:
            angle_error = 360 - angle_error

        if angle_error > ANGLE_TOLERANCE:
            continue

        # Require the candidate to be on the correct side of the
        # predicted runway center.
        candidate_top_x = line_x_at_y(a, b, c, d, y1)
        candidate_bottom_x = line_x_at_y(a, b, c, d, height - 1)

        if candidate_top_x is None or candidate_bottom_x is None:
            continue

        if side == "left":
            if candidate_bottom_x >= width / 2:
                continue
        else:
            if candidate_bottom_x <= width / 2:
                continue

        top_distance = abs(candidate_top_x - x1)
        if top_distance > TOP_DISTANCE:
            continue

        length = np.hypot(c - a, d - b)

        # Prefer long, strong-looking lines that stay close to the
        # predicted vanishing/top point.
        score = angle_error * 4.0 + top_distance - length * 0.08

        if score < best_score:
            best_score = score
            best = (a, b, c, d)

    if best is None:
        return (x1, y1, x2, y2)

    a, b, c, d = best

    # Preserve the YOLO top point, but use the image-derived boundary
    # direction and extend it to the bottom of the frame.
    if abs(c - a) < 1:
        return (x1, y1, x1, height - 1)

    slope = (d - b) / (c - a)
    bottom_x = x1 + slope * ((height - 1) - y1)

    bottom_x = max(0, min(width - 1, bottom_x))

    return (x1, y1, bottom_x, height - 1)


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
    refined_count = 0

    print()
    print("========== RUNWAY VIDEO DEMO V3 ==========")
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

                left = refine_boundary(frame, lt, lb, "left")
                right = refine_boundary(frame, rt, rb, "right")

                current_lines = np.array([left, right], dtype=np.float32)

                if left != (lt[0], lt[1], lb[0], height - 1):
                    refined_count += 1

                detected_count += 1

        if current_lines is not None:
            if previous_lines is None:
                smoothed = current_lines
            else:
                smoothed = (
                    SMOOTHING * previous_lines
                    + (1.0 - SMOOTHING) * current_lines
                )

            previous_lines = smoothed

            left = tuple(smoothed[0].astype(int))
            right = tuple(smoothed[1].astype(int))

            cv2.line(frame, left[:2], left[2:], (0, 255, 0), 4)
            cv2.line(frame, right[:2], right[2:], (0, 255, 0), 4)

            center_top = (
                (left[0] + right[0]) // 2,
                (left[1] + right[1]) // 2
            )
            center_bottom = (
                (left[2] + right[2]) // 2,
                (left[3] + right[3]) // 2
            )

            cv2.line(
                frame, center_top, center_bottom,
                (255, 0, 0), 3
            )

            cv2.putText(
                frame, "RUNWAY DETECTED",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 3
            )

        elif previous_lines is not None:
            left = tuple(previous_lines[0].astype(int))
            right = tuple(previous_lines[1].astype(int))

            cv2.line(frame, left[:2], left[2:], (0, 255, 255), 4)
            cv2.line(frame, right[:2], right[2:], (0, 255, 255), 4)

        writer.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed frames: {frame_count}")

    cap.release()
    writer.release()

    print()
    print("========== DEMO V3 COMPLETE ==========")
    print("Total frames:", frame_count)
    print("Frames with 4 YOLO keypoints:", detected_count)
    print("Frames using image-edge refinement:", refined_count)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
