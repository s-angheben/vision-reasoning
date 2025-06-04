from model import QwenVLModel
import json
import os
import re
import random
import string
from caltech101 import Caltech101


DATASET_PATH = "/home/samuele.angheben/datasets"
BASE_PATH = "/home/samuele.angheben/vision-reasoning/qwen_caltech_set"

dataset = Caltech101(root=DATASET_PATH, download=True, split='test', transform=None)

print("Loaded dataset with categories:", dataset.categories)

model = QwenVLModel()
print("Model loaded.")


prompts = {
    "prompt1": ("Identify the object. Use 1 to 3 words.", False),
    "prompt2": ("Label the primary object (max 3 words).", False),
    "prompt3": ("What is this? Provide a 1-3 word description.", False),
    "prompt4": ("Classify the object. Use 1 to 3 words.", False),
    "prompt5": ("Open-world classification: Name the object in 1 to 3 words.", False),
    # Additional prompts for open-world classification behavior:
    "prompt6": ("Act as an image classifier. What is the main object? Respond with 1-3 words.", False),
    "prompt7": ("You are a classifier. Give the object class in 1 to 3 words.", False),
    "prompt8": ("Classify the main object in this image using up to 3 words.", False),
    "prompt9": ("Provide the category of the object in 1-3 words, as a classifier would.", False),
    "prompt10": ("As an open-world classifier, state the object's class (max 3 words).", False),
    "prompt11": ("What is that? Use 1 to 3 words.", False),
}

def normalize_text(text):
    """Normalize text by replacing punctuation with spaces and converting to lowercase"""
    # Replace punctuation with spaces, then normalize multiple spaces to single spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    return ' '.join(text.split())  # Remove extra whitespace

def label_in_prediction(label, prediction):
    """Check if prediction contains the label words with simple matching logic"""
    normalized_label = normalize_text(label)
    normalized_prediction = normalize_text(prediction)
    
    # Split into words for better word boundary matching
    label_words = normalized_label.split()
    pred_words = normalized_prediction.split()
    
    # For single word labels, check if that word exists in prediction
    if len(label_words) == 1:
        return label_words[0] in pred_words
    
    # For multi-word labels, check if all words exist in prediction
    return all(word in pred_words for word in label_words)


# Initial step: test all prompts on 4 random images and print predictions
""" print("\n=== Initial prompt testing on 4 random images ===")
sample_indices = random.sample(range(len(dataset)), 4)
for idx in sample_indices:
    image, label = dataset[idx]
    label_name = dataset.categories[label]
    print(f"\nImage idx: {idx}, Label: {label_name}")
    for prompt_name, (prompt_text, is_reasoning) in prompts.items():
        prediction = model.predict(image, prompt_text)
        if is_reasoning:
            match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
            if match:
                answer = match.group(1).strip()
            else:
                answer = ""
            print(f"[{prompt_name}] Reasoning prediction: {prediction} | Extracted answer: '{answer}'")
        else:
            print(f"[{prompt_name}] Prediction: {prediction}") """


os.makedirs(f"{BASE_PATH}/outputs", exist_ok=True)

category_outputs_all = {cat: set() for cat in dataset.categories}

for prompt_name, (prompt_text, is_reasoning) in prompts.items():
    print(f"Processing {prompt_name}: '{prompt_text}' (reasoning={is_reasoning})")
    output_file = f"{BASE_PATH}/outputs/predictions_{prompt_name}.txt"
    
    category_outputs = {cat: set() for cat in dataset.categories}
    invalid_count = 0  # Track invalid outputs for reasoning prompts
    correct = 0
    total = 0

    with open(output_file, "w") as f:
        for idx, (image, label) in enumerate(dataset):
            prediction = model.predict(image, prompt_text)
            if is_reasoning:
                # Extract content inside <answer>...</answer>
                match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
                if match:
                    prediction = match.group(1).strip()
                else:
                    prediction = ""
                    invalid_count += 1
                    print(f"[{prompt_name}] Invalid reasoning output at example {idx}")
            ground_truth = dataset.categories[label]
            # Add prediction to the set for the ground truth category
            category_outputs[ground_truth].add(prediction)
            # Add prediction to the global category_outputs_all
            category_outputs_all[ground_truth].add(prediction)

            # Accuracy calculation
            total += 1
            is_correct = label_in_prediction(ground_truth, prediction)
            if is_correct:
                correct += 1
            
            # Print with correctness indicator
            status = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"[{prompt_name}] Example {idx}: label={dataset.categories[label]}, prediction={prediction} [{status}]")


        # Convert sets to lists for JSON serialization
        serializable_outputs = {cat: list(outputs) for cat, outputs in category_outputs.items()}
        accuracy = correct / total if total > 0 else 0.0
        if is_reasoning:
            output_data = {
                "category_outputs": serializable_outputs,
                "invalid_count": invalid_count,
                "accuracy": accuracy
            }
            json.dump(output_data, f, indent=2)
        else:
            output_data = {
                "category_outputs": serializable_outputs,
                "accuracy": accuracy
            }
            json.dump(output_data, f, indent=2)
    print(f"Saved predictions to {output_file}")

# After all prompts, save category_outputs_all to a file
category_outputs_all_serializable = {cat: list(outputs) for cat, outputs in category_outputs_all.items()}
all_output_file = f"{BASE_PATH}/outputs/category_outputs_all.json"
with open(all_output_file, "w") as f:
    json.dump(category_outputs_all_serializable, f, indent=2)
print(f"Saved all category outputs to {all_output_file}")



