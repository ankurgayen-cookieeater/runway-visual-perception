from pathlib import Path

ROOT = Path(r".\final_yolo_pose")

SPLITS = ["train", "val", "test"]


def main():
    total_images = 0
    total_labels = 0
    errors = 0

    print()
    print("========== FINAL DATASET CHECK ==========")

    for split in SPLITS:
        images = list((ROOT / "images" / split).glob("*.jpg"))
        labels = list((ROOT / "labels" / split).glob("*.txt"))

        image_names = {x.stem for x in images}
        label_names = {x.stem for x in labels}

        missing_labels = image_names - label_names
        missing_images = label_names - image_names

        split_errors = len(missing_labels) + len(missing_images)

        print()
        print(split.upper())
        print("Images:", len(images))
        print("Labels:", len(labels))
        print("Errors:", split_errors)

        errors += split_errors
        total_images += len(images)
        total_labels += len(labels)

    print()
    print("========== SUMMARY ==========")
    print("Total images:", total_images)
    print("Total labels:", total_labels)
    print("Total errors:", errors)

    if errors == 0:
        print("RESULT: DATASET PASSED")
    else:
        print("RESULT: DATASET FAILED")


if __name__ == "__main__":
    main()