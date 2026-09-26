import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL = r".\runs\pose\train-7\weights\best.pt"
VIDEO = r".\UAV\simulation\runway_02.mp4"
CSV_FILE = r".\UAV\ground_truth_sidelines_extraction\outputrunway_02.mp4.csv"

FRAMES = [605, 630, 680, 730, 756]

OUTPUT_DIR = Path(r".\train7_visual_check")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_ground_truth(csv_file):
    import csv
    import ast

    ground_truth = {}

    with open(csv_file, "r", newline="") as f:
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
            except:
                continue

    return ground_truth


def draw_points(frame, points, color, label):
    for i, (x, y) in enumerate(points):
        x = int(x)
        y = int(y)

        cv2.circle(frame, (x, y), 6, color, -1)
        cv2.putText(
            frame,
            f"{label}{i}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

def draw_lines(frame, points, color):
    if len(points) != 4:
        return

    p0 = tuple(map(int, points[0]))
    p1 = tuple(map(int, points[1]))
    p2 = tuple(map(int, points[2]))
    p3 = tuple(map(int, points[3]))

    cv2.line(frame, p0, p1, color, 3)
    cv2.line(frame, p2, p3, color, 3)

def main():
    print()
    print("========== TRAIN-7 VISUAL CHECK ==========")

    model = YOLO(MODEL)
    ground_truth = load_ground_truth(CSV_FILE)

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    for frame_number in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)

        success, frame = cap.read()

        if not success:
            print("Could not read frame:", frame_number)
            continue

        results = model.predict(
            frame,
            imgsz=640,
            device=0,
            verbose=False
        )

        result = results[0]

        # Draw ground truth in red
        if frame_number in ground_truth:
            gt = ground_truth[frame_number]

            draw_points(frame, gt, (0, 0, 255), "GT")
            draw_lines(frame, gt, (0, 0, 255))

        # Draw prediction in green
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            confidences = result.boxes.conf

            best_index = int(confidences.argmax())

            predicted = result.keypoints.xy[best_index].cpu().numpy()

            draw_points(
                frame,
                predicted,
                (0, 255, 0),
                "P"
            )

            draw_lines(
                frame,
                predicted,
                (0, 255, 0)
            )

            confidence = float(confidences[best_index])

            cv2.putText(
                frame,
                f"Confidence: {confidence:.3f}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        output_file = OUTPUT_DIR / f"frame_{frame_number:04d}.jpg"

        cv2.imwrite(str(output_file), frame)

        print("Saved:", output_file)

    cap.release()

    print()
    print("========== VISUAL CHECK COMPLETE ==========")
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()