# Test qwen2.5VL 2b model on the Caltech-UCSD Birds 200-2011 dataset
from dataset import CUB200Dataset
from model import QwenVLModel
import os
import datetime
import re

# Base path configuration
BASE_PATH = "/home/samuele.angheben/vision-reasoning/qwen_bird"

def evaluate_dataset(dataset, dataset_name, output_file, prompt, model, class_names_dict, is_reasoning=False):
    """Evaluate a dataset and save results to file"""
    correct = 0
    total = 0
    
    def normalize_text(text):
        """Normalize text by replacing punctuation with spaces and converting to lowercase"""
        # Replace punctuation with spaces, then normalize multiple spaces to single spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        return ' '.join(text.split())  # Remove extra whitespace
    
    def extract_answer_from_tags(text):
        """Extract text between <answer></answer> tags"""
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def check_accuracy(ground_truth, prediction):
        """Check if prediction contains the ground truth with simple logic"""
        normalized_gt = normalize_text(ground_truth)
        normalized_pred = normalize_text(prediction)
        
        # Simple substring check
        return normalized_gt in normalized_pred

    with open(output_file, "w") as f:
        f.write(f"{dataset_name} Dataset Predictions - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        for idx, sample in enumerate(dataset):
            prediction = model.predict(sample["image"], prompt)
            ground_truth = class_names_dict[sample['label']]
            
            # Extract answer from tags if using reasoning prompt
            if is_reasoning:
                prediction_for_check = extract_answer_from_tags(prediction)
                # If no answer tags found, mark as incorrect
                if prediction_for_check is None:
                    is_correct = False
                else:
                    is_correct = check_accuracy(ground_truth, prediction_for_check)
            else:
                prediction_for_check = prediction
                is_correct = check_accuracy(ground_truth, prediction_for_check)
            
            if is_correct:
                correct += 1
            total += 1
            
            current_accuracy = correct / total

            print(f"Sample {idx}: Ground truth: {ground_truth}")
            print(f"Sample {idx}: Prediction: {prediction}")
            print(f"Sample {idx}: Correct: {is_correct}")
            print(f"Sample {idx}: Running accuracy: {current_accuracy:.4f}")
            print("---")
            
            f.write(f"Sample {idx}:\n")
            f.write(f"Ground truth: {ground_truth}\n")
            f.write(f"Prediction: {prediction}\n")
            f.write(f"Correct: {is_correct}\n")
            f.write(f"Running accuracy: {current_accuracy:.4f}\n")
            f.write("-" * 40 + "\n\n")
        
        f.write(f"\n{dataset_name} dataset accuracy: {correct}/{total} = {correct/total:.4f}\n")
    
    return correct, total

CUB200Dataset = CUB200Dataset(split='test')

model = QwenVLModel()
prompt = f"Please identify the bird species in this image. Choose from the following list of bird species:\n\n{CUB200Dataset.prompt_class_list}\n\nProvide your answer as the species name."
reasoning_prompt = f"""You are an expert ornithologist. Carefully analyze the visual features of the bird in the image (such as color, size, beak shape, markings, and other distinctive traits). 

Step 1: Describe the key visual features you observe.
Step 2: Based on these features, select the most likely species from the following list:

{CUB200Dataset.prompt_class_list}

Step 3: Clearly state your final answer by writing only the species name inside <answer></answer> tags.

Begin your reasoning below:
"""

# Create output directory
os.makedirs(f"{BASE_PATH}/outputs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Evaluate both datasets
output_file_original = f"{BASE_PATH}/outputs/predictions_original_{timestamp}.txt"
correct_original, total_original = evaluate_dataset(
    CUB200Dataset.get_dataset(), "Original", output_file_original, 
    prompt, model, CUB200Dataset.class_names_dict
)

output_file_original_reasoning = f"{BASE_PATH}/outputs/predictions_original_reasoning_{timestamp}.txt"
correct_original_reasoning, total_original_reasoning = evaluate_dataset(
    CUB200Dataset.get_dataset(), "Original Reasoning", output_file_original_reasoning, 
    reasoning_prompt, model, CUB200Dataset.class_names_dict, is_reasoning=True
)

output_file_cropped = f"{BASE_PATH}/outputs/predictions_cropped_{timestamp}.txt"
correct_cropped, total_cropped = evaluate_dataset(
    CUB200Dataset.get_dataset_cropped(), "Cropped", output_file_cropped, 
    prompt, model, CUB200Dataset.class_names_dict
)

output_file_cropped_reasoning = f"{BASE_PATH}/outputs/predictions_cropped_reasoning_{timestamp}.txt"
correct_cropped_reasoning, total_cropped_reasoning = evaluate_dataset(
    CUB200Dataset.get_dataset_cropped(), "Cropped Reasoning", output_file_cropped_reasoning, 
    reasoning_prompt, model, CUB200Dataset.class_names_dict, is_reasoning=True
)

print(f"Original dataset accuracy: {correct_original}/{total_original} = {correct_original/total_original:.4f}")
print(f"Original reasoning accuracy: {correct_original_reasoning}/{total_original_reasoning} = {correct_original_reasoning/total_original_reasoning:.4f}")
print(f"Cropped dataset accuracy: {correct_cropped}/{total_cropped} = {correct_cropped/total_cropped:.4f}")
print(f"Cropped reasoning accuracy: {correct_cropped_reasoning}/{total_cropped_reasoning} = {correct_cropped_reasoning/total_cropped_reasoning:.4f}")

# Save summary results
summary_file = f"{BASE_PATH}/outputs/accuracy_summary_{timestamp}.txt"
with open(summary_file, "w") as f:
    f.write(f"Bird Classification Accuracy Summary\n")
    f.write("=" * 40 + "\n\n")
    f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: Qwen2.5-VL-3B-Instruct\n")
    f.write(f"Dataset: Caltech-UCSD Birds 200-2011 (test set)\n\n")
    f.write(f"Original Dataset:\n")
    f.write(f"  Correct: {correct_original}/{total_original}\n")
    f.write(f"  Accuracy: {correct_original/total_original:.4f} ({correct_original/total_original*100:.2f}%)\n\n")
    f.write(f"Original Dataset (Reasoning):\n")
    f.write(f"  Correct: {correct_original_reasoning}/{total_original_reasoning}\n")
    f.write(f"  Accuracy: {correct_original_reasoning/total_original_reasoning:.4f} ({correct_original_reasoning/total_original_reasoning*100:.2f}%)\n\n")
    f.write(f"Cropped Dataset:\n")
    f.write(f"  Correct: {correct_cropped}/{total_cropped}\n")
    f.write(f"  Accuracy: {correct_cropped/total_cropped:.4f} ({correct_cropped/total_cropped*100:.2f}%)\n\n")
    f.write(f"Cropped Dataset (Reasoning):\n")
    f.write(f"  Correct: {correct_cropped_reasoning}/{total_cropped_reasoning}\n")
    f.write(f"  Accuracy: {correct_cropped_reasoning/total_cropped_reasoning:.4f} ({correct_cropped_reasoning/total_cropped_reasoning*100:.2f}%)\n\n")

print(f"Summary saved to: {summary_file}")

