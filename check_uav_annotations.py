import os
import glob

VIDEO_DIRS = [
    r".\UAV\real_videos",
    r".\UAV\simulation"
]

GT_DIR = r".\UAV\ground_truth_sidelines_extraction"


def main():
    videos = []

    for folder in VIDEO_DIRS:
        videos.extend(
            glob.glob(os.path.join(folder, "*.mp4"))
        )
        videos.extend(
            glob.glob(os.path.join(folder, "*.MP4"))
        )

    csv_files = glob.glob(
        os.path.join(GT_DIR, "*.csv")
    )

    print()
    print("========== UAV ANNOTATION INVENTORY ==========")

    print()
    print("VIDEOS:")
    for video in videos:
        print(os.path.basename(video))

    print()
    print("GROUND-TRUTH CSV FILES:")
    for csv_file in csv_files:
        print(os.path.basename(csv_file))

    print()
    print("Video count:", len(videos))
    print("CSV count:", len(csv_files))

    print()
    print("========== COMPLETE ==========")


if __name__ == "__main__":
    main()