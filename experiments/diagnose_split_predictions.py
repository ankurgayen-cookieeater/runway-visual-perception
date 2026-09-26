from ultralytics import YOLO
import cv2
import os


MODEL = r".\runs\pose\train\weights\best.pt"

VIDEO = r".\UAV\simulation\runway_02.mp4"

OUTPUT_DIR = r".\yolo_pose_split_diagnostics"


SAMPLES = {
    "train": [1, 200, 453],
    "val": [454, 530, 604],
    "test": [605, 680, 730, 756]
}


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(MODEL)

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    print()
    print("========== YOLO POSE SPLIT DIAGNOSTIC ==========")

    for split, frame_numbers in SAMPLES.items():

        split_dir = os.path.join(
            OUTPUT_DIR,
            split
        )

        os.makedirs(
            split_dir,
            exist_ok=True
        )

        print()
        print("----------", split.upper(), "----------")

        for frame_number in frame_numbers:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                frame_number - 1
            )

            ret, frame = cap.read()

            if not ret:
                print(
                    "Could not read frame:",
                    frame_number
                )
                continue

            results = model(
                frame,
                device=0,
                conf=0.001,
                verbose=False
            )

            result = results[0]

            print(
                "Frame:",
                frame_number,
                "| Detections:",
                len(result.boxes)
            )

            for i in range(len(result.boxes)):

                confidence = float(
                    result.boxes.conf[i].cpu().item()
                )

                print(
                    "  Detection:",
                    i,
                    "| Confidence:",
                    f"{confidence:.6f}"
                )

            annotated = result.plot()

            output_path = os.path.join(
                split_dir,
                f"frame_{frame_number:04d}.jpg"
            )

            cv2.imwrite(
                output_path,
                annotated
            )

            print(
                "  Saved:",
                output_path
            )

    cap.release()

    print()
    print("========== DIAGNOSTIC COMPLETE ==========")
    print(
        "Results saved to:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()