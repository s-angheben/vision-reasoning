import json
import os

# Define file paths
base_dir = os.path.dirname(__file__)
files_to_merge = [
    "category_outputs_all.json",
    "category_outputs_generic_all.json",
    "category_outputs_specific_all.json"
]
output_file = "category_all.json"

# Load and merge JSON contents
merged = []
for fname in files_to_merge:
    path = os.path.join(base_dir, fname)
    with open(path, "r") as f:
        data = json.load(f)
        if isinstance(data, list):
            merged.extend(data)
        elif isinstance(data, dict):
            merged.append(data)
        else:
            raise ValueError(f"Unsupported JSON type in {fname}")

# Write merged output
with open(os.path.join(base_dir, output_file), "w") as f:
    json.dump(merged, f, indent=2)
