import cv2
import csv
import ast
import shutil
from pathlib import Path

ROOT = Path(r".\UAV")
OUT = Path(r".\final_yolo_pose")

SEQUENCES = {
    "train": [
        (
            "real_videos",
            "GX010028Trim2.mp4",
            "outputVideoGX010028Trim2.csv"
        ),
    ],

    "val": [
        (
            "real_videos",
            "GX010035Trim1.mp4",
            "outpuGX010035Trim1.csv"
        ),
    ],

    "test": [
        (
            "simulation",
            "runway_02.mp4",
            "outputrunway_02.mp4.csv"
        ),
    ],
}


def read_annotations(csv_path):
    annotations = {}

    with open(csv_path, "r") as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue

            frame = int(row[1])
            points = ast.literal_eval(row[2])
            annotations[frame] = points

    return annotations


def main():

    if OUT.exists():
        shutil.rmtree(OUT)

    for split in SEQUENCES:
        (OUT / "images" / split).mkdir(
            parents=True,
            exist_ok=True
        )

        (OUT / "labels" / split).mkdir(
            parents=True,
            exist_ok=True
        )

    counts = {
        "train": 0,
        "val": 0,
        "test": 0
    }

    for split, sequences in SEQUENCES.items():

        for folder, video_name, csv_name in sequences:

            video_path = ROOT / folder / video_name
            csv_path = (
                ROOT
                / "ground_truth_sidelines_extraction"
                / csv_name
            )

            annotations = read_annotations(csv_path)

            cap = cv2.VideoCapture(str(video_path))

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            print()
            print(split.upper(), video_name)
            print("Annotated frames:", len(annotations))
            print(
                "Resolution:",
                width,
                "x",
                height
            )

            if not cap.isOpened():
                print("ERROR: Could not open video.")
                cap.release()
                continue

            for frame_number, points in annotations.items():

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_number - 1
                )

                ok, frame = cap.read()

                if not ok:
                    continue

                xs = [p[0] for p in points]
                ys = [p[1] for p in points]

                xmin = min(xs)
                xmax = max(xs)
                ymin = min(ys)
                ymax = max(ys)

                xc = ((xmin + xmax) / 2) / width
                yc = ((ymin + ymax) / 2) / height
                bw = (xmax - xmin) / width
                bh = (ymax - ymin) / height

                keypoints = []

                for x, y in points:
                    keypoints.extend([
                        x / width,
                        y / height
                    ])

                label = [
                    0,
                    xc,
                    yc,
                    bw,
                    bh,
                    *keypoints
                ]

                name = (
                    f"{Path(video_name).stem}"
                    f"_frame_{frame_number:04d}"
                )

                image_path = (
                    OUT
                    / "images"
                    / split
                    / f"{name}.jpg"
                )

                label_path = (
                    OUT
                    / "labels"
                    / split
                    / f"{name}.txt"
                )

                cv2.imwrite(
                    str(image_path),
                    frame
                )

                with open(label_path, "w") as f:
                    f.write(
                        " ".join(
                            f"{value:.6f}"
                            for value in label
                        )
                    )

                counts[split] += 1

            cap.release()

    print()
    print("========== FINAL DATASET ==========")
    print("Train:", counts["train"])
    print("Validation:", counts["val"])
    print("Test:", counts["test"])
    print("Total:", sum(counts.values()))
    print("Saved to:", OUT)


if __name__ == "__main__":
    main()