import csv
import ast

CSV_FILE = r".\UAV\ground_truth_sidelines_extraction\outputrunway_02.mp4.csv"

frame_numbers = []

with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)

    for row in reader:
        if len(row) < 3:
            continue

        frame_number = int(row[1])
        points = ast.literal_eval(row[2])

        frame_numbers.append(frame_number)

print("Annotated frames:", len(frame_numbers))
print("First annotated frame:", min(frame_numbers))
print("Last annotated frame:", max(frame_numbers))

print()
print("First 10 frames:", frame_numbers[:10])
print("Last 10 frames:", frame_numbers[-10:])