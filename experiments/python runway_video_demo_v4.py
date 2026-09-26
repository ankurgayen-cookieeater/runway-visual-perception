import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

MODEL = r".\runs\pose\train-4\weights\best.pt"
VAL_DIR = Path(r".\final_yolo_pose\images\val")
VAL_LABEL_DIR = Path(r".\final_yolo_pose\labels\val")
VIDEO = r".\UAV\simulation\runway_02.mp4"
OUTPUT = r".\outputs\runway_demo_v4.mp4"


def load_gt(label_path, width, height):
    values = list(map(float, label_path.read_text().split()))
    pts = np.array(values[5:13], dtype=np.float32).reshape(4, 2)
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts


def main():
    model = YOLO(MODEL)

    # Learn a simple affine correction from the validation sequence only.
    X = [[] for _ in range(4)]
    Y = [[] for _ in range(4)]

    val_images = sorted(VAL_DIR.glob("*.jpg"))
    for image_path in val_images:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        h, w = image.shape[:2]
        result = model.predict(image, imgsz=640, device=0, conf=0.25, verbose=False)[0]

        if result.keypoints is None or len(result.keypoints.xy) == 0:
            continue

        pred = result.keypoints.xy[0].cpu().numpy()
        if len(pred) != 4:
            continue

        gt = load_gt(VAL_LABEL_DIR / f"{image_path.stem}.txt", w, h)

        for i in range(4):
            # [pred_x, pred_y, 1] -> corrected [gt_x, gt_y]
            X[i].append([pred[i, 0], pred[i, 1], 1.0])
            Y[i].append(gt[i])

    corrections = []
    for i in range(4):
        A = np.asarray(X[i], dtype=np.float64)
        B = np.asarray(Y[i], dtype=np.float64)
        M, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
        corrections.append(M)

    print()
    print("========== RUNWAY VIDEO DEMO V4 ==========")
    print("Validation samples used:", len(X[0]))
    print("Learned keypoint correction from validation set.")

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

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            frame, imgsz=640, device=0, conf=0.25, verbose=False
        )[0]

        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            pred = result.keypoints.xy[0].cpu().numpy()

            if len(pred) == 4:
                corrected = []

                for i in range(4):
                    row = np.array([pred[i, 0], pred[i, 1], 1.0])
                    point = row @ corrections[i]
                    corrected.append(point)

                pts = np.asarray(corrected, dtype=np.float32)
                pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

                lb, lt, rb, rt = [tuple(p.astype(int)) for p in pts]

                cv2.line(frame, lb, lt, (0, 255, 0), 4)
                cv2.line(frame, rb, rt, (0, 255, 0), 4)

                center_top = ((lt[0] + rt[0]) // 2, (lt[1] + rt[1]) // 2)
                center_bottom = ((lb[0] + rb[0]) // 2, (lb[1] + rb[1]) // 2)

                cv2.line(frame, center_top, center_bottom, (255, 0, 0), 3)

                for point in (lb, lt, rb, rt):
                    cv2.circle(frame, point, 5, (0, 255, 0), -1)

                cv2.putText(
                    frame, "RUNWAY DETECTED",
                    (30, 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 3
                )

                detected_count += 1

        writer.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"Processed frames: {frame_count}")

    cap.release()
    writer.release()

    print()
    print("========== DEMO V4 COMPLETE ==========")
    print("Total frames:", frame_count)
    print("Frames with 4 keypoints:", detected_count)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
