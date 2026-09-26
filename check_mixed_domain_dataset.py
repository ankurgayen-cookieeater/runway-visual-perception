from pathlib import Path

ROOT = Path(r"..\final_yolo_pose_mixed_domain")

SPLITS = ["train", "val", "test"]


def main():
    print()
    print("========== MIXED-DOMAIN DATASET CHECK ==========")

    total_errors = 0

    for split in SPLITS:
        image_dir = ROOT / "images" / split
        label_dir = ROOT / "labels" / split

        images = sorted(image_dir.glob("*.jpg"))
        labels = sorted(label_dir.glob("*.txt"))

        image_names = {p.stem for p in images}
        label_names = {p.stem for p in labels}

        errors = 0

        if image_names != label_names:
            missing_labels = image_names - label_names
            missing_images = label_names - image_names

            for name in sorted(missing_labels):
                print(f"ERROR: Missing label: {split}/{name}.txt")
                errors += 1

            for name in sorted(missing_images):
                print(f"ERROR: Missing image: {split}/{name}.jpg")
                errors += 1

        for label_file in labels:
            try:
                lines = label_file.read_text(encoding="utf-8").strip().splitlines()

                if len(lines) != 1:
                    print(f"ERROR: Wrong line count: {label_file}")
                    errors += 1
                    continue

                values = lines[0].split()

                if len(values) != 13:
                    print(
                        f"ERROR: Wrong value count: "
                        f"{label_file} -> {len(values)}"
                    )
                    errors += 1
                    continue

                numbers = [float(v) for v in values]

                if int(numbers[0]) != 0:
                    print(f"ERROR: Wrong class: {label_file}")
                    errors += 1

                for value in numbers[1:]:
                    if not 0.0 <= value <= 1.0:
                        print(
                            f"ERROR: Value outside [0,1]: "
                            f"{label_file}"
                        )
                        errors += 1
                        break

            except Exception as e:
                print(f"ERROR: Could not read {label_file}: {e}")
                errors += 1

        print()
        print(f"---------- {split.upper()} ----------")
        print("Images:", len(images))
        print("Labels:", len(labels))
        print("Errors:", errors)

        total_errors += errors

    print()
    print("========== SUMMARY ==========")
    print("Total errors:", total_errors)

    if total_errors == 0:
        print("RESULT: DATASET PASSED")
    else:
        print("RESULT: DATASET FAILED")


if __name__ == "__main__":
    main()