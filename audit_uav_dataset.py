from pathlib import Path
import csv
import cv2
import re

ROOT = Path(r"D:\python_projects\YOLO_projects\aviation_cv\UAV")
OUT = ROOT.parent / "uav_dataset_inventory.csv"

VIDEO_EXTS = {".mp4", ".MP4", ".avi", ".mov", ".MOV", ".mkv", ".MKV"}

def read_csv_info(csv_path):
    rows = []
    try:
        with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                video = row[0].strip()
                try:
                    frame = int(row[1].strip())
                except ValueError:
                    continue
                rows.append((video, frame))
    except Exception as e:
        return None, str(e)

    if not rows:
        return None, "No annotation rows"

    videos = sorted(set(v for v, _ in rows))
    frames = [fr for _, fr in rows]
    return {
        "video_names": videos,
        "annotated_frames": len(rows),
        "first_frame": min(frames),
        "last_frame": max(frames),
    }, ""

def main():
    print()
    print("========== UAV DATASET MASTER AUDIT ==========")
    print("Root:", ROOT)

    if not ROOT.exists():
        print("ERROR: UAV root does not exist.")
        return

    video_files = sorted(
        p for p in ROOT.rglob("*")
        if p.is_file() and p.suffix in VIDEO_EXTS
    )

    csv_files = sorted(
        p for p in ROOT.rglob("*.csv")
        if p.is_file()
    )

    print("Video files found:", len(video_files))
    print("CSV files found:", len(csv_files))
    print()

    csv_info = {}
    for csv_path in csv_files:
        info, error = read_csv_info(csv_path)
        if info:
            csv_info[csv_path] = info
            print(
                "CSV:",
                csv_path.relative_to(ROOT),
                "| annotations:", info["annotated_frames"],
                "| frames:", info["first_frame"], "-", info["last_frame"],
                "| video(s):", ", ".join(info["video_names"])
            )
        else:
            print("CSV:", csv_path.relative_to(ROOT), "| ERROR:", error)

    print()
    print("---------- VIDEO AUDIT ----------")

    # Map video basenames to files.
    video_map = {}
    for p in video_files:
        video_map.setdefault(p.name.lower(), []).append(p)

    rows = []

    for i, video_path in enumerate(video_files, 1):
        cap = cv2.VideoCapture(str(video_path))
        opened = cap.isOpened()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else -1
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else -1
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else -1
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else -1

        if opened:
            cap.release()

        matching_csvs = []
        for csv_path, info in csv_info.items():
            if video_path.name.lower() in [v.lower() for v in info["video_names"]]:
                matching_csvs.append(csv_path)

        row = {
            "video": video_path.name,
            "relative_path": str(video_path.relative_to(ROOT)),
            "readable": opened,
            "width": width,
            "height": height,
            "fps": fps,
            "video_frames": frame_count,
            "matching_csv_count": len(matching_csvs),
            "matching_csvs": ";".join(str(p.relative_to(ROOT)) for p in matching_csvs),
            "annotated_frames": sum(csv_info[p]["annotated_frames"] for p in matching_csvs),
            "first_annotated_frame": min(
                (csv_info[p]["first_frame"] for p in matching_csvs),
                default=-1
            ),
            "last_annotated_frame": max(
                (csv_info[p]["last_frame"] for p in matching_csvs),
                default=-1
            ),
        }
        rows.append(row)

        status = "READABLE" if opened else "UNREADABLE"
        print(
            f"[{i}/{len(video_files)}] {status} | "
            f"{video_path.relative_to(ROOT)} | "
            f"{width}x{height} | FPS {fps:.2f} | "
            f"frames {frame_count} | "
            f"GT {row['annotated_frames']}"
        )

    fields = [
        "video", "relative_path", "readable", "width", "height", "fps",
        "video_frames", "matching_csv_count", "matching_csvs",
        "annotated_frames", "first_annotated_frame", "last_annotated_frame"
    ]

    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    readable = sum(r["readable"] for r in rows)
    annotated = sum(r["annotated_frames"] > 0 for r in rows)
    matched_csvs = sum(r["matching_csv_count"] > 0 for r in rows)

    print()
    print("========== AUDIT SUMMARY ==========")
    print("Videos:", len(rows))
    print("Readable videos:", readable)
    print("Unreadable videos:", len(rows) - readable)
    print("Videos with matching ground truth:", matched_csvs)
    print("Videos with annotated frames:", annotated)
    print("Inventory saved to:", OUT)
    print()
    print("IMPORTANT: No training data was changed.")
    print("This audit only inventories the UAV dataset.")
    print("========== AUDIT COMPLETE ==========")

if __name__ == "__main__":
    main()
