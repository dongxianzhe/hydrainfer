import os
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--folder", type=str, default="result")

args = parser.parse_args()

folder = args.folder

pattern = re.compile(r"(.+?):\s+([\d.]+)")

metrics = []

for root, _, files in os.walk(folder):
    for file in files:
        if file.endswith("result.log"):
            file_path = os.path.join(root, file)

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    match = pattern.search(line)
                    if match:
                        key = match.group(1).strip()
                        value = match.group(2).strip()
                        metrics.append((file, key, value))

from collections import defaultdict

results = defaultdict(list)

for file, key, value in metrics:
    results[key].append((file, value))

for metric, values in results.items():
    print(f"\n{metric}:")
    sorted_values = sorted(values, key=lambda x: x[0])
    for file, value in sorted_values:
        print(f"  {file}: {value}")