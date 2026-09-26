import cv2
import csv
import ast
import shutil
from pathlib import Path


ROOT = Path(r"D:\python_projects\YOLO_projects\aviation_cv")

OUTPUT = ROOT / "final_yolo_pose_mixed_domain"

VIDEOS = {
    "GX010028Trim2": ROOT / "UAV" / "real_videos" / "GX010028Trim2.mp4",
    "runway_video20230228-103640": ROOT / "UAV" / "simulation" / "runway_video20230228-103640.mp4",
    "GX010035Trim1": ROOT / "UAV" / "real_videos" / "GX010035Trim1.mp4",
    "runway_02": ROOT / "UAV" / "simulation" / "runway_02.mp4",
}

CSV_FILES = {
    "GX010028Trim2": ROOT / "UAV" / "ground_truth_sidelines_extraction" / "outputVideoGX010028Trim2.csv",
    "runway_video20230228-103640": ROOT / "UAV" / "ground_truth_sidelines_extraction" / "outputrunway_video20230228-103640.csv",
    "GX010035Trim1": ROOT / "UAV" / "ground_truth_sidelines_extraction" / "outpuGX010035Trim1.csv",
    "runway_02": ROOT / "UAV" / "ground_truth_sidelines_extraction" / "outputrunway_02.mp4.csv",
}

SPLITS = {
    "train": ["GX010028Trim2", "runway_video20230228-103640"],
    "val": ["GX010035Trim1"],
    "test": ["runway_02"],
}


def read_annotations(csv_file):
    annotations = {}

    with open(csv_file, "r", newline="") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 3:
                continue

            try:
                frame_number = int(row[1])
                points_text = ",".join(row[2:]).strip()
                points = ast.literal_eval(points_text)

                if len(points) != 4:
                    continue

                points = [(float(x), float(y)) for x, y in points]
                annotations[frame_number] = points

            except (ValueError, SyntaxError):
                continue

    return annotations


def make_label(points, width, height):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    x_min = max(0.0, min(xs))
    x_max = min(float(width), max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(float(height), max(ys))

    x_center = ((x_min + x_max) / 2.0) / width
    y_center = ((y_min + y_max) / 2.0) / height
    box_width = (x_max - x_min) / width
    box_height = (y_max - y_min) / height

    values = [
        0,
        x_center,
        y_center,
        box_width,
        box_height,
    ]

    for x, y in points:
        values.extend([x / width, y / height])

    return " ".join(f"{v:.6f}" for v in values)


def process_video(video_key, split):
    video_file = VIDEOS[video_key]
    csv_file = CSV_FILES[video_key]

    annotations = read_annotations(csv_file)

    cap = cv2.VideoCapture(str(video_file))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    image_dir = OUTPUT / "images" / split
    label_dir = OUTPUT / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for frame_number in sorted(annotations):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        success, frame = cap.read()

        if not success:
            print(f"WARNING: could not read {video_key} frame {frame_number}")
            continue

        points = annotations[frame_number]

        image_name = f"{video_key}_frame_{frame_number:04d}.jpg"
        label_name = f"{video_key}_frame_{frame_number:04d}.txt"

        cv2.imwrite(str(image_dir / image_name), frame)

        label = make_label(points, width, height)
        (label_dir / label_name).write_text(label, encoding="utf-8")

        saved += 1

    cap.release()

    print(f"{split.upper():5s} {video_key}: {saved} annotated frames saved")


def write_yaml():
    yaml_text = f"""path: {OUTPUT.as_posix()}

train: images/train
val: images/val
test: images/test

kpt_shape: [4, 2]

flip_idx: [2, 3, 0, 1]

names:
  0: runway
"""

    (OUTPUT / "runway_pose.yaml").write_text(yaml_text, encoding="utf-8")


def main():
    print()
    print("========== MIXED-DOMAIN POSE DATASET ==========")

    if OUTPUT.exists():
        print("Removing previous mixed-domain dataset...")
        shutil.rmtree(OUTPUT)

    for split, video_keys in SPLITS.items():
        for video_key in video_keys:
            process_video(video_key, split)

    write_yaml()

    counts = {}

    for split in SPLITS:
        images = list((OUTPUT / "images" / split).glob("*.jpg"))
        labels = list((OUTPUT / "labels" / split).glob("*.txt"))
        counts[split] = len(images)

        print(
            f"{split.upper():5s}: "
            f"{len(images)} images, {len(labels)} labels"
        )

    print()
    print("========== DATASET COMPLETE ==========")
    print("Train:", counts["train"])
    print("Validation:", counts["val"])
    print("Test:", counts["test"])
    print("Total:", sum(counts.values()))
    print("Dataset:", OUTPUT)
    print("YAML:", OUTPUT / "runway_pose.yaml")


if __name__ == "__main__":
    main()
