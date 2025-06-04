from torchvision.datasets import Caltech101
from model import QwenVLModel
import json
import os
import re

DATASET_PATH = "/home/samuele.angheben/datasets"
BASE_PATH = "/home/samuele.angheben/vision-reasoning/qwen_caltech_set"

dataset = Caltech101(root=DATASET_PATH, download=True, transform=None)

exit(0)
model = QwenVLModel()

prompts = {
    "prompt1": ("What type of object is in this photo?", False),
    "prompt2": ("What type of object is in this photo? Be generic.", False),
    "prompt3": ("What type of object is in this photo? Be specific.", False),
    "prompt4": ("What type of object is in this photo? Think step by step and give the final answer in <answer> </answer> tags", True),
    "prompt5": ("What type of object is in this photo? Please output the thinking process in <think> </think> and final answer in <answer> </answer> tags", True),
}

os.makedirs(f"{BASE_PATH}/outputs", exist_ok=True)

category_outputs_all = {cat: set() for cat in dataset.categories}

for prompt_name, (prompt_text, is_reasoning) in prompts.items():
    output_file = f"{BASE_PATH}/outputs/predictions_{prompt_name}.txt"
    
    category_outputs = {cat: set() for cat in dataset.categories}
    invalid_count = 0  # Track invalid outputs for reasoning prompts

    with open(output_file, "w") as f:
        for (image, label) in dataset:
            prediction = model.predict(image, prompt_text)
            if is_reasoning:
                # Extract content inside <answer>...</answer>
                match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
                if match:
                    prediction = match.group(1).strip()
                else:
                    prediction = ""
                    invalid_count += 1
            ground_truth = dataset.categories[label]
            # Add prediction to the set for the ground truth category
            category_outputs[ground_truth].add(prediction)
            # Add prediction to the global category_outputs_all
            category_outputs_all[ground_truth].add(prediction)

        # Convert sets to lists for JSON serialization
        serializable_outputs = {cat: list(outputs) for cat, outputs in category_outputs.items()}
        if is_reasoning:
            output_data = {
                "category_outputs": serializable_outputs,
                "invalid_count": invalid_count
            }
            json.dump(output_data, f, indent=2)
        else:
            json.dump(serializable_outputs, f, indent=2)

# After all prompts, save category_outputs_all to a file
category_outputs_all_serializable = {cat: list(outputs) for cat, outputs in category_outputs_all.items()}
with open(f"{BASE_PATH}/outputs/category_outputs_all.json", "w") as f:
    json.dump(category_outputs_all_serializable, f, indent=2)



