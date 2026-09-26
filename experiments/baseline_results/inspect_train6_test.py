import cv2
from pathlib import Path
from ultralytics import YOLO


MODEL = r"..\runs\pose\train-6\weights\best.pt"
VIDEO = r"..\UAV\simulation\runway_02.mp4"
CSV_FILE = r"..\UAV\ground_truth_sidelines_extraction\outputrunway_02.mp4.csv"

FRAMES = [605, 630, 680, 730, 756]

OUTPUT_DIR = Path(r".\train6_visual_check")
OUTPUT_DIR.mkdir(exist_ok=True)


def read_ground_truth():
    import ast
    import csv

    ground_truth = {}

    with open(CSV_FILE, "r", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 3:
                continue

            try:
                frame = int(row[1])
                points_text = ",".join(row[2:]).strip()
                points = ast.literal_eval(points_text)

                if len(points) == 4:
                    ground_truth[frame] = points

            except (ValueError, SyntaxError):
                continue

    return ground_truth


def main():

    print()
    print("========== TRAIN-6 VISUAL CHECK ==========")

    model = YOLO(MODEL)
    ground_truth = read_ground_truth()

    cap = cv2.VideoCapture(VIDEO)

    for frame_number in FRAMES:

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

        success, frame = cap.read()

        if not success:
            print("Could not read frame:", frame_number)
            continue

        results = model.predict(
            frame,
            imgsz=640,
            conf=0.001,
            device=0,
            verbose=False
        )

        result = results[0]

        # Draw ground truth in RED
        if frame_number in ground_truth:

            points = ground_truth[frame_number]

            for x, y in points:
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    7,
                    (0, 0, 255),
                    -1
                )

            # Left runway boundary
            cv2.line(
                frame,
                tuple(map(int, points[0])),
                tuple(map(int, points[1])),
                (0, 0, 255),
                3
            )

            # Right runway boundary
            cv2.line(
                frame,
                tuple(map(int, points[2])),
                tuple(map(int, points[3])),
                (0, 0, 255),
                3
            )

        # Draw Train-6 prediction in GREEN
        if result.keypoints is not None and len(result.keypoints) > 0:

            best_index = 0

            if result.boxes is not None and len(result.boxes) > 1:
                best_index = int(result.boxes.conf.argmax())

            keypoints = result.keypoints.xy[best_index].cpu().numpy()

            confidence = float(
                result.boxes.conf[best_index].cpu().item()
            )

            for x, y in keypoints:
                cv2.circle(
                    frame,
                    (int(x), int(y)),
                    7,
                    (0, 255, 0),
                    -1
                )

            if len(keypoints) == 4:

                # Left predicted boundary
                cv2.line(
                    frame,
                    tuple(map(int, keypoints[0])),
                    tuple(map(int, keypoints[1])),
                    (0, 255, 0),
                    3
                )

                # Right predicted boundary
                cv2.line(
                    frame,
                    tuple(map(int, keypoints[2])),
                    tuple(map(int, keypoints[3])),
                    (0, 255, 0),
                    3
                )

            cv2.putText(
                frame,
                f"Train-6 confidence: {confidence:.3f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO DETECTION",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"Frame: {frame_number}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        output_file = OUTPUT_DIR / f"frame_{frame_number:04d}.jpg"

        cv2.imwrite(str(output_file), frame)

        print("Saved:", output_file)

    cap.release()

    print()
    print("========== VISUAL CHECK COMPLETE ==========")


if __name__ == "__main__":
    main()