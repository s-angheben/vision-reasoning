# Test qwen2.5VL 2b model on the Caltech-UCSD Birds 200-2011 dataset
from dataset import CUB200Dataset
from model import QwenVLModel
import os
import datetime
import json

# Base path configuration
BASE_PATH = "/home/samuele.angheben/vision-reasoning/qwen_bird_open"

CUB200Dataset = CUB200Dataset(split='test')
dataset = CUB200Dataset.get_dataset()
class_names_dict = CUB200Dataset.class_names_dict

model = QwenVLModel()
prompt = "Analyze the given image and predict the most specific and accurate label possible for the primary object or scene depicted. Use scientific or technical terms when applicable to enhance specificity. If there is uncertainty about the exact label, provide a more general category or abstain from making a prediction. Ensure that all predictions are accurate and avoid guessing. The response should only contain the possible classification label, limited to a maximum of 1-3 words."

os.makedirs(f"{BASE_PATH}/outputs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

results = []

for idx, sample in enumerate(dataset):
    prediction = model.predict(sample["image"], prompt)
    ground_truth = class_names_dict[sample['label']]
    results.append({
        "index": idx,
        "prediction": prediction,
        "ground_truth": ground_truth
    })

output_path = f"{BASE_PATH}/outputs/results_{timestamp}.json"
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

