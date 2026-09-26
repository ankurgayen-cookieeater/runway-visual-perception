from ultralytics import YOLO
import cv2
import os

MODEL_640 = r".\runs\pose\train-2\weights\best.pt"
MODEL_960 = r".\runs\pose\train-3\weights\best.pt"

IMAGE_DIR = r".\yolo_runway_pose\images\test"
OUTPUT_DIR = r".\comparison_640_960"

FRAMES = [660, 670, 680, 690, 700, 710, 720, 730, 740, 750]


def draw_predictions(image, results, title):
    output = image.copy()

    if len(results) > 0:
        result = results[0]

        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            keypoints = result.keypoints.xy.cpu().numpy()

            if result.boxes is not None:
                confidences = result.boxes.conf.cpu().numpy()
            else:
                confidences = []

            for det_idx, points in enumerate(keypoints):
                conf = confidences[det_idx] if det_idx < len(confidences) else 0

                # Draw the four keypoints
                for x, y in points:
                    cv2.circle(
                        output,
                        (int(x), int(y)),
                        5,
                        (255, 0, 0),
                        -1
                    )

                # Connect runway keypoints
                if len(points) >= 4:
                    cv2.line(
                        output,
                        (int(points[0][0]), int(points[0][1])),
                        (int(points[1][0]), int(points[1][1])),
                        (255, 0, 0),
                        3
                    )

                    cv2.line(
                        output,
                        (int(points[2][0]), int(points[2][1])),
                        (int(points[3][0]), int(points[3][1])),
                        (255, 0, 0),
                        3
                    )

                    cv2.line(
                        output,
                        (int(points[0][0]), int(points[0][1])),
                        (int(points[2][0]), int(points[2][1])),
                        (255, 0, 0),
                        2
                    )

                    cv2.line(
                        output,
                        (int(points[1][0]), int(points[1][1])),
                        (int(points[3][0]), int(points[3][1])),
                        (255, 0, 0),
                        2
                    )

                cv2.putText(
                    output,
                    f"conf={conf:.3f}",
                    (20, 40 + det_idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2
                )

    cv2.putText(
        output,
        title,
        (20, 680),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    return output


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print()
    print("========== 640 vs 960 VISUAL COMPARISON ==========")

    print("Loading 640px model...")
    model_640 = YOLO(MODEL_640)

    print("Loading 960px model...")
    model_960 = YOLO(MODEL_960)

    for frame_number in FRAMES:

        image_path = os.path.join(
            IMAGE_DIR,
            f"frame_{frame_number:04d}.jpg"
        )

        image = cv2.imread(image_path)

        if image is None:
            print("Could not read:", image_path)
            continue

        result_640 = model_640.predict(
            source=image,
            imgsz=640,
            device=0,
            verbose=False
        )

        result_960 = model_960.predict(
            source=image,
            imgsz=960,
            device=0,
            verbose=False
        )

        left = draw_predictions(
            image,
            result_640,
            f"640px | Frame {frame_number}"
        )

        right = draw_predictions(
            image,
            result_960,
            f"960px | Frame {frame_number}"
        )

        comparison = cv2.hconcat([left, right])

        output_path = os.path.join(
            OUTPUT_DIR,
            f"compare_{frame_number:04d}.jpg"
        )

        cv2.imwrite(output_path, comparison)

        print("Saved:", output_path)

    print()
    print("========== COMPARISON COMPLETE ==========")
    print("Output folder:", OUTPUT_DIR)


if __name__ == "__main__":
    main()