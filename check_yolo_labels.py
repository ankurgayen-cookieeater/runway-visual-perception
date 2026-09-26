import os


DATASET_ROOT = r".\yolo_runway_pose"

SPLITS = [
    "train",
    "val",
    "test"
]

TOLERANCE = 1e-5


def main():

    total_files = 0
    total_errors = 0

    print()
    print("========== YOLO LABEL INTEGRITY CHECK ==========")

    for split in SPLITS:

        label_dir = os.path.join(
            DATASET_ROOT,
            "labels",
            split
        )

        files = sorted(
            filename
            for filename in os.listdir(label_dir)
            if filename.endswith(".txt")
        )

        split_errors = 0

        print()
        print("----------", split.upper(), "----------")
        print("Label files:", len(files))

        for filename in files:

            path = os.path.join(
                label_dir,
                filename
            )

            with open(path, "r") as file:

                lines = [
                    line.strip()
                    for line in file
                    if line.strip()
                ]

            if len(lines) != 1:

                print(
                    "ERROR:",
                    filename,
                    "| Expected 1 line, found:",
                    len(lines)
                )

                split_errors += 1
                continue

            values = lines[0].split()

            if len(values) != 13:

                print(
                    "ERROR:",
                    filename,
                    "| Expected 13 values, found:",
                    len(values)
                )

                split_errors += 1
                continue

            try:

                values = [
                    float(value)
                    for value in values
                ]

            except ValueError:

                print(
                    "ERROR:",
                    filename,
                    "| Contains non-numeric values"
                )

                split_errors += 1
                continue

            class_id = int(values[0])

            if class_id != 0:

                print(
                    "ERROR:",
                    filename,
                    "| Invalid class:",
                    class_id
                )

                split_errors += 1

            coordinates = values[1:]

            for value in coordinates:

                if value < -TOLERANCE or value > 1 + TOLERANCE:

                    print(
                        "ERROR:",
                        filename,
                        "| Coordinate outside [0,1]:",
                        value
                    )

                    split_errors += 1
                    break

            bbox_x = values[1]
            bbox_y = values[2]
            bbox_width = values[3]
            bbox_height = values[4]

            if bbox_width <= 0:

                print(
                    "ERROR:",
                    filename,
                    "| Invalid bbox width:",
                    bbox_width
                )

                split_errors += 1

            if bbox_height <= 0:

                print(
                    "ERROR:",
                    filename,
                    "| Invalid bbox height:",
                    bbox_height
                )

                split_errors += 1

            bbox_x_min = (
                bbox_x -
                bbox_width / 2
            )

            bbox_x_max = (
                bbox_x +
                bbox_width / 2
            )

            bbox_y_min = (
                bbox_y -
                bbox_height / 2
            )

            bbox_y_max = (
                bbox_y +
                bbox_height / 2
            )

            keypoints = []

            for i in range(5, 13, 2):

                keypoints.append(
                    (
                        values[i],
                        values[i + 1]
                    )
                )

            for x, y in keypoints:

                if (
                    x < bbox_x_min - TOLERANCE
                    or
                    x > bbox_x_max + TOLERANCE
                ):

                    print(
                        "ERROR:",
                        filename,
                        "| Keypoint x outside bbox"
                    )

                    split_errors += 1
                    break

                if (
                    y < bbox_y_min - TOLERANCE
                    or
                    y > bbox_y_max + TOLERANCE
                ):

                    print(
                        "ERROR:",
                        filename,
                        "| Keypoint y outside bbox"
                    )

                    split_errors += 1
                    break

        total_files += len(files)
        total_errors += split_errors

        print("Errors:", split_errors)

    print()
    print("========== SUMMARY ==========")
    print("Total label files:", total_files)
    print("Total errors:", total_errors)

    if total_errors == 0:

        print("RESULT: ALL LABELS PASSED")

    else:

        print("RESULT: LABEL ERRORS FOUND")

    print()
    print("Label integrity check complete.")


if __name__ == "__main__":
    main()