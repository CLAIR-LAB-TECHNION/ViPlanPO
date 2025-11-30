import json
import pandas as pd
import os

# Path to your JSONL log file
input_path = os.path.expanduser("~/execution_2025-11-30_16-18-43.log")
output_path = os.path.expanduser("~/execution_2025-11-30_16-18-43.csv")

records = []

# Read JSONL lines
with open(input_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            records.append(obj)
        except json.JSONDecodeError:
            print("Skipping invalid JSON line:", line)

# Convert to DataFrame
df = pd.DataFrame(records)

# Save to CSV
df.to_csv(output_path, index=False)

print("CSV saved to:", output_path)
