import cv2
from pathlib import Path
from ultralytics import YOLO


MODEL = r".\runs\pose\train-7\weights\best.pt"
VIDEO = r".\UAV\simulation\runway_02.mp4"

OUTPUT_DIR = Path(r".\outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "runway_train7_final_demo.mp4"


def draw_point(frame, point, color, label):
    x, y = map(int, point)

    cv2.circle(frame, (x, y), 6, color, -1)

    cv2.putText(
        frame,
        label,
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


def draw_line(frame, p1, p2, color, thickness=4):
    p1 = tuple(map(int, p1))
    p2 = tuple(map(int, p2))

    cv2.line(
        frame,
        p1,
        p2,
        color,
        thickness
    )


def main():

    print()
    print("========== FINAL RUNWAY DEMO ==========")

    model = YOLO(MODEL)

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Video width:", width)
    print("Video height:", height)
    print("FPS:", fps)
    print("Total frames:", total_frames)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video.")

    processed = 0
    detected = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        processed += 1

        results = model.predict(
            frame,
            imgsz=640,
            device=0,
            verbose=False
        )

        result = results[0]

        if (
            result.keypoints is not None
            and len(result.keypoints.xy) > 0
        ):

            confidences = result.boxes.conf

            best_index = int(confidences.argmax())
            confidence = float(confidences[best_index])

            points = result.keypoints.xy[
                best_index
            ].cpu().numpy()

            if len(points) == 4:

                detected += 1

                # Keypoint order:
                # 0 = left-bottom
                # 1 = left-top
                # 2 = right-bottom
                # 3 = right-top

                p0 = points[0]
                p1 = points[1]
                p2 = points[2]
                p3 = points[3]

                # Green runway boundaries
                draw_line(
                    frame,
                    p0,
                    p1,
                    (0, 255, 0),
                    4
                )

                draw_line(
                    frame,
                    p2,
                    p3,
                    (0, 255, 0),
                    4
                )

                # Green keypoints
                draw_point(
                    frame,
                    p0,
                    (0, 255, 0),
                    "LB"
                )

                draw_point(
                    frame,
                    p1,
                    (0, 255, 0),
                    "LT"
                )

                draw_point(
                    frame,
                    p2,
                    (0, 255, 0),
                    "RB"
                )

                draw_point(
                    frame,
                    p3,
                    (0, 255, 0),
                    "RT"
                )

                # Runway centerline
                center_bottom = (
                    (p0[0] + p2[0]) / 2,
                    (p0[1] + p2[1]) / 2
                )

                center_top = (
                    (p1[0] + p3[0]) / 2,
                    (p1[1] + p3[1]) / 2
                )

                draw_line(
                    frame,
                    center_bottom,
                    center_top,
                    (255, 0, 0),
                    3
                )

                cv2.putText(
                    frame,
                    f"Runway detected | Confidence: {confidence:.3f}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Runway keypoints unavailable",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2
                )

        else:

            cv2.putText(
                frame,
                "Runway not detected",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 255),
                2
            )

        cv2.putText(
            frame,
            f"Frame: {processed}/{total_frames}",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        writer.write(frame)

        if processed % 100 == 0:
            print(
                f"Processed: {processed}/{total_frames}"
            )

    cap.release()
    writer.release()

    print()
    print("========== DEMO COMPLETE ==========")
    print("Processed frames:", processed)
    print("Frames with runway prediction:", detected)
    print("Output:", OUTPUT_VIDEO)


if __name__ == "__main__":
    main()