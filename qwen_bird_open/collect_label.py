# Test qwen2.5VL 2b model on the Caltech-UCSD Birds 200-2011 dataset
from dataset import CUB200Dataset
from model import QwenVLModel
import os
import datetime
import json

# Base path configuration
BASE_PATH = "/home/samuele.angheben/vision-reasoning/qwen_bird_open"

CUB200Dataset = CUB200Dataset(split='train')
dataset = CUB200Dataset.get_dataset()
class_names_dict = CUB200Dataset.class_names_dict

model = QwenVLModel()
prompt = "For open world classification task, analyze the given image and predict the most specific and accurate label possible for the primary object or scene depicted. Use scientific or technical terms when applicable to enhance specificity. If there is uncertainty about the exact label, provide a more general category or abstain from making a prediction. Avoid wrong predictions. Ensure that all predictions are accurate and avoid guessing. The response should only contain the possible classification label, limited to a maximum of 1-3 words."

outputs_dir = os.path.join(BASE_PATH, "outputs_train")
os.makedirs(outputs_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

results = []
class_predictions = {}

for idx, sample in enumerate(dataset):
    print(f"Processing sample {idx} / {len(dataset)}")
    predictions = model.predict_multiple(
        sample["image"], prompt,
        do_sample=True,
        top_k=100,
        top_p=0.95,
        temperature=1.3,
        num_return_sequences=100,
        max_new_tokens=64
    )
    print(f"Predictions for sample {idx}: {predictions}")
    ground_truth = class_names_dict[sample['label']]
    print(f"Ground truth for sample {idx}: {ground_truth}")
    results.append({
        "index": idx,
        "predictions": predictions,
        "ground_truth": ground_truth
    })
    # Group predictions by ground truth class
    if ground_truth not in class_predictions:
        class_predictions[ground_truth] = []
    class_predictions[ground_truth].extend(predictions)

output_path = os.path.join(outputs_dir, f"predictions_{timestamp}.json")
print(f"Saving all predictions to {output_path}")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

# Save class-wise predictions
class_pred_path = os.path.join(outputs_dir, f"perclass_predictions_{timestamp}.json")
print(f"Saving class-wise predictions to {class_pred_path}")
with open(class_pred_path, "w") as f:
    json.dump(class_predictions, f, indent=2)