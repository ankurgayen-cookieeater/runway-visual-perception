import csv
import glob
import os

for file in glob.glob(r".\UAV\ground_truth_sidelines_extraction\*.csv"):
    frames = []

    with open(file, "r") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                try:
                    frames.append(int(row[1]))
                except ValueError:
                    pass

    print(os.path.basename(file))
    print("  annotated frames:", len(frames))
    print("  first frame:", min(frames))
    print("  last frame:", max(frames))
    print()