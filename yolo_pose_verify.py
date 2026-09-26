import cv2
import os


DATASET_ROOT = r".\yolo_runway_pose"

SAMPLES = {
    "train": [1, 200, 453],
    "val": [454, 530, 604],
    "test": [605, 680, 756]
}


def read_label(label_path):

    with open(label_path, "r") as file:
        line = file.readline().strip()

    values = list(map(float, line.split()))

    class_id = int(values[0])

    bbox = values[1:5]

    keypoints = []

    for i in range(5, len(values), 2):
        keypoints.append(
            (values[i], values[i + 1])
        )

    return class_id, bbox, keypoints


def main():

    output_root = r".\yolo_pose_verification"

    os.makedirs(output_root, exist_ok=True)

    for split, frame_numbers in SAMPLES.items():

        image_dir = os.path.join(
            DATASET_ROOT,
            "images",
            split
        )

        label_dir = os.path.join(
            DATASET_ROOT,
            "labels",
            split
        )

        split_output_dir = os.path.join(
            output_root,
            split
        )

        os.makedirs(
            split_output_dir,
            exist_ok=True
        )

        for frame_number in frame_numbers:

            filename = f"frame_{frame_number:04d}.jpg"

            image_path = os.path.join(
                image_dir,
                filename
            )

            label_path = os.path.join(
                label_dir,
                f"frame_{frame_number:04d}.txt"
            )

            image = cv2.imread(image_path)

            if image is None:
                print("Could not read image:", image_path)
                continue

            class_id, bbox, keypoints = read_label(label_path)

            height, width = image.shape[:2]

            bbox_x, bbox_y, bbox_width, bbox_height = bbox

            x_center = bbox_x * width
            y_center = bbox_y * height

            box_width = bbox_width * width
            box_height = bbox_height * height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)

            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            pixel_points = []

            for x, y in keypoints:

                px = int(x * width)
                py = int(y * height)

                pixel_points.append(
                    (px, py)
                )

                cv2.circle(
                    image,
                    (px, py),
                    8,
                    (0, 0, 255),
                    -1
                )

            if len(pixel_points) == 4:

                cv2.line(
                    image,
                    pixel_points[0],
                    pixel_points[1],
                    (255, 0, 0),
                    3
                )

                cv2.line(
                    image,
                    pixel_points[2],
                    pixel_points[3],
                    (255, 0, 0),
                    3
                )

            output_path = os.path.join(
                split_output_dir,
                filename
            )

            cv2.imwrite(
                output_path,
                image
            )

            print(
                "Saved:",
                output_path,
                "| Class:",
                class_id,
                "| Keypoints:",
                len(keypoints)
            )

    print()
    print("========== YOLO POSE VERIFICATION ==========")
    print("Verification images saved to:", output_root)


if __name__ == "__main__":
    main()