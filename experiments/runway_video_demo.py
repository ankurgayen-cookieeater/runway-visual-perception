import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL = r".\runs\pose\train-4\weights\best.pt"
VIDEO = r".\UAV\simulation\runway_02.mp4"
OUTPUT = r".\outputs\runway_demo.mp4"


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

    frame_count = 0
    detected_count = 0

    print()
    print("========== RUNWAY VIDEO DEMO ==========")
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
            verbose=False
        )
        result = results[0]

        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            points = result.keypoints.xy[0].cpu().numpy()

            if len(points) == 4:
                detected_count += 1

                lb, lt, rb, rt = [tuple(p.astype(int)) for p in points]

                cv2.line(frame, lb, lt, (0, 255, 0), 3)
                cv2.line(frame, rb, rt, (0, 255, 0), 3)

                center_bottom = ((lb[0] + rb[0]) // 2,
                                 (lb[1] + rb[1]) // 2)
                center_top = ((lt[0] + rt[0]) // 2,
                              (lt[1] + rt[1]) // 2)

                cv2.line(frame, center_bottom, center_top, (255, 0, 0), 3)

                for point, label in zip(
                    (lb, lt, rb, rt), ("LB", "LT", "RB", "RT")
                ):
                    cv2.circle(frame, point, 6, (0, 255, 0), -1)
                    cv2.putText(
                        frame, label, (point[0] + 8, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                    )

                cv2.putText(
                    frame, "RUNWAY DETECTED", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3
                )

        writer.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed frames: {frame_count}")

    cap.release()
    writer.release()

    print()
    print("========== DEMO COMPLETE ==========")
    print("Total frames:", frame_count)
    print("Frames with 4 keypoints:", detected_count)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
