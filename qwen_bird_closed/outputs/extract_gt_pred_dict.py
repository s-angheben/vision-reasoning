import re
from collections import defaultdict

# Path to the predictions file
pred_file = "predictions_original_20250529_230219.txt"

# Dict: ground truth -> set of predictions
gt_to_preds = defaultdict(set)

with open(pred_file, "r", encoding="utf-8") as f:
    gt = None
    for line in f:
        if line.startswith("Ground truth:"):
            gt = line.strip().split("Ground truth:")[1].strip()
        elif line.startswith("Prediction:"):
            pred = line.strip().split("Prediction:")[1].strip()
            if gt is not None:
                gt_to_preds[gt].add(pred)
                gt = None  # Reset for next sample

# Example: print the dictionary
for gt, preds in gt_to_preds.items():
    print(f"{gt}: {preds}")
