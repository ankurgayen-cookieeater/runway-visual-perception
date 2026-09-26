from pathlib import Path
import ast
import csv
import cv2
import shutil

ROOT = Path(r"D:\python_projects\YOLO_projects\aviation_cv\UAV")
OUT = ROOT.parent / "final_yolo_pose_all_gt"

SEQUENCES = {
    "train": [
        ("real_videos", "GX010028Trim2.mp4", "outputVideoGX010028Trim2.csv"),
        ("real_videos", "GX010035Trim1.mp4", "outpuGX010035Trim1.csv"),
    ],
    "val": [
        ("simulation", "runway_video20230228-103640.mp4",
         "outputrunway_video20230228-103640.csv"),
    ],
    "test": [
        ("simulation", "runway_02.mp4", "outputrunway_02.mp4.csv"),
    ],
}

def read_annotations(csv_path):
    annotations = {}

    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 3:
                continue

            try:
                frame_id = int(row[1].strip())
            except ValueError:
                continue

            # The coordinate list contains commas, so CSV parsing splits
            # it across multiple columns. Rejoin everything after column 1.
            points_text = ",".join(row[2:]).strip()

            try:
                points = ast.literal_eval(points_text)
            except Exception:
                continue

            if not isinstance(points, (list, tuple)) or len(points) != 4:
                continue

            try:
                points = [
                    (float(p[0]), float(p[1]))
                    for p in points
                ]
            except Exception:
                continue

            annotations[frame_id] = points

    return annotations

def save_sample(frame, points, image_path, label_path, width, height):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    xmin = max(0.0, min(xs))
    xmax = min(float(width - 1), max(xs))
    ymin = max(0.0, min(ys))
    ymax = min(float(height - 1), max(ys))

    if xmax <= xmin or ymax <= ymin:
        return False

    values = [
        0.0,
        ((xmin + xmax) / 2.0) / width,
        ((ymin + ymax) / 2.0) / height,
        (xmax - xmin) / width,
        (ymax - ymin) / height,
    ]

    for x, y in points:
        values.extend([x / width, y / height])

    if not cv2.imwrite(str(image_path), frame):
        return False

    label_path.write_text(
        " ".join(f"{v:.6f}" for v in values) + "\n",
        encoding="utf-8"
    )

    return True

def process_split(split, sequences):
    image_dir = OUT / "images" / split
    label_dir = OUT / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    total = 0

    for folder, video_name, csv_name in sequences:
        video_path = ROOT / folder / video_name
        csv_path = ROOT / "ground_truth_sidelines_extraction" / csv_name

        print()
        print(f"========== {split.upper()} ==========")
        print("Video:", video_path)
        print("CSV:", csv_path)

        annotations = read_annotations(csv_path)

        print("Annotated frames:", len(annotations))

        if not annotations:
            raise RuntimeError(
                f"CSV parser found 0 annotations in:\n{csv_path}"
            )

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("Resolution:", width, "x", height)
        print("Video frames:", frame_count)

        saved = 0
        current = 0
        stem = Path(video_name).stem

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            current += 1

            if current not in annotations:
                continue

            image_path = image_dir / f"{stem}_{current:06d}.jpg"
            label_path = label_dir / f"{stem}_{current:06d}.txt"

            if save_sample(
                frame,
                annotations[current],
                image_path,
                label_path,
                width,
                height
            ):
                saved += 1
                total += 1

        cap.release()

        print("Saved:", saved)

        if saved != len(annotations):
            raise RuntimeError(
                f"Expected {len(annotations)} frames but saved {saved} "
                f"for {video_name}"
            )

    return total

def write_yaml():
    yaml = f"""path: {OUT.as_posix()}

train: images/train
val: images/val
test: images/test

kpt_shape: [4, 2]

flip_idx: [2, 3, 0, 1]

names:
  0: runway
"""
    (OUT / "runway_pose.yaml").write_text(yaml, encoding="utf-8")

def main():
    print()
    print("========== FULL LABELED UAV DATASET BUILD ==========")

    # Automatically remove the empty/previous failed build.
    if OUT.exists():
        print("Removing previous dataset build...")
        shutil.rmtree(OUT)

    print("Output:", OUT)

    totals = {}

    for split, sequences in SEQUENCES.items():
        totals[split] = process_split(split, sequences)

    write_yaml()

    print()
    print("========== FINAL DATASET ==========")
    print("Train:", totals["train"])
    print("Validation:", totals["val"])
    print("Test:", totals["test"])
    print("Total:", sum(totals.values()))
    print("Dataset:", OUT)
    print("YAML:", OUT / "runway_pose.yaml")
    print()
    print("========== DATASET BUILD COMPLETE ==========")

if __name__ == "__main__":
    main()
