from ultralytics import YOLO
import cv2


MODEL = r".\runs\pose\train-2\weights\best.pt"

IMAGE = r".\yolo_runway_pose\images\test\frame_0605.jpg"

LABEL = r".\yolo_runway_pose\labels\test\frame_0605.txt"


def main():

    print()
    print("========== YOLO TEST IMAGE INSPECTION ==========")

    model = YOLO(MODEL)

    image = cv2.imread(IMAGE)

    if image is None:
        raise RuntimeError(
            "Could not read image: " + IMAGE
        )

    print("Image:", IMAGE)
    print("Image shape:", image.shape)

    with open(LABEL, "r") as file:

        label_values = list(
            map(
                float,
                file.readline().split()
            )
        )

    print()
    print("Ground-truth label:")
    print(label_values)

    results = model(
        image,
        device=0,
        conf=0.001,
        verbose=False
    )

    result = results[0]

    print()
    print("Detections:", len(result.boxes))

    for i in range(len(result.boxes)):

        confidence = float(
            result.boxes.conf[i].cpu().item()
        )

        class_id = int(
            result.boxes.cls[i].cpu().item()
        )

        box = result.boxes.xyxy[i].cpu().numpy()

        print()
        print("Detection:", i)
        print(
            "Confidence:",
            f"{confidence:.6f}"
        )
        print(
            "Class:",
            class_id
        )
        print(
            "Box:",
            box
        )

        if result.keypoints is not None:

            keypoints = (
                result.keypoints.xy[i]
                .cpu()
                .numpy()
            )

            print("Keypoints:")

            for j, point in enumerate(keypoints):

                print(
                    f"  {j}: "
                    f"({point[0]:.2f}, "
                    f"{point[1]:.2f})"
                )

    output = result.plot()

    output_path = r".\yolo_test_image_0605.jpg"

    cv2.imwrite(
        output_path,
        output
    )

    print()
    print("Visualization saved to:")
    print(output_path)

    print()
    print("Inspection complete.")


if __name__ == "__main__":
    main()