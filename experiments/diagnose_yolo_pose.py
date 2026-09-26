from ultralytics import YOLO
import cv2
import os


MODEL = r".\runs\pose\train\weights\best.pt"

VIDEO = r".\UAV\simulation\runway_02.mp4"

OUTPUT_DIR = r".\yolo_pose_diagnostics"

FRAMES = [605, 630, 680, 730, 756]


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(MODEL)

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    print()
    print("========== YOLO POSE DIAGNOSTIC ==========")

    for frame_number in FRAMES:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number - 1
        )

        ret, frame = cap.read()

        if not ret:
            print("Could not read frame:", frame_number)
            continue

        results = model(
            frame,
            device=0,
            conf=0.001,
            verbose=False
        )

        result = results[0]

        print()
        print("Frame:", frame_number)
        print("Detections:", len(result.boxes))

        if len(result.boxes) == 0:
            print("No detections even at conf=0.001")
            continue

        for i in range(len(result.boxes)):

            confidence = float(
                result.boxes.conf[i].cpu().item()
            )

            class_id = int(
                result.boxes.cls[i].cpu().item()
            )

            print(
                "Detection:",
                i,
                "| Class:",
                class_id,
                "| Confidence:",
                f"{confidence:.6f}"
            )

            if result.keypoints is not None:

                keypoints = result.keypoints.xy[i].cpu().numpy()

                print(
                    "Keypoints:",
                    keypoints
                )

        annotated = result.plot()

        output_path = os.path.join(
            OUTPUT_DIR,
            f"frame_{frame_number:04d}.jpg"
        )

        cv2.imwrite(
            output_path,
            annotated
        )

        print("Saved:", output_path)

    cap.release()

    print()
    print("========== DIAGNOSTIC COMPLETE ==========")
    print("Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()