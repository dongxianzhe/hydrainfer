import re
from collections import defaultdict

import argparse

parser = argparse.ArgumentParser(description="读取路径参数示例")
parser.add_argument("--log-path", type=str, required=True)
args = parser.parse_args()

pattern = re.compile(r"(\w+)\s+time:\s+([\d.e-]+)")

time_data = defaultdict(list)

with open(args.log_path, "r") as file:
    for line in file:
        match = pattern.search(line)
        if match:
            param, time = match.groups()
            time_data[param].append(float(time))

if time_data:
    print(f"{'stage name':<20} {'time':<15} {'number requests':<10} {'avg time (s)'}")
    print("=" * 55)
    
    for param, times in time_data.items():
        total_time = sum(times)
        avg_time = total_time / len(times)
        print(f"{param:<20} {total_time:<15.6f} {len(times):<10} {avg_time:.6f}")
else:
    print('no latency data captured')