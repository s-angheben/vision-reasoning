# Test qwen2.5VL 2b model on the Caltech-UCSD Birds 200-2011 dataset
from dataset import CUB200Dataset
from model import QwenVLModel
import os
import datetime

# Base path configuration
BASE_PATH = "/home/samuele.angheben/vision-reasoning"

def evaluate_dataset(dataset, dataset_name, output_file, prompt, model, class_names_dict, progress_file):
    """Evaluate a dataset and save results to file"""
    correct = 0
    total = 0
    
    with open(output_file, "w") as f, open(progress_file, "w") as pf:
        f.write(f"{dataset_name} Dataset Predictions - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        pf.write(f"{dataset_name} Progressive Accuracy - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        pf.write("Sample,Correct,Total,Accuracy\n")
        
        for idx, sample in enumerate(dataset):
            prediction = model.predict(sample["image"], prompt)
            ground_truth = class_names_dict[sample['label']]
            
            # Simple accuracy check - if ground truth is in prediction
            is_correct = ground_truth.lower() in prediction.lower()
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
            f.write("-" * 40 + "\n\n")
            
            pf.write(f"{idx},{correct},{total},{current_accuracy:.4f}\n")
        
        f.write(f"\n{dataset_name} dataset accuracy: {correct}/{total} = {correct/total:.4f}\n")
    
    return correct, total

CUB200Dataset = CUB200Dataset(split='test')

model = QwenVLModel()
prompt = f"Please identify the bird species in this image. Choose from the following list of bird species:\n\n{CUB200Dataset.prompt_class_list}\n\nProvide your answer as the species name."

# Create output directory
os.makedirs(f"{BASE_PATH}/outputs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Evaluate both datasets
output_file_original = f"{BASE_PATH}/outputs/predictions_original_{timestamp}.txt"
progress_file_original = f"{BASE_PATH}/outputs/progress_original_{timestamp}.csv"
correct_original, total_original = evaluate_dataset(
    CUB200Dataset.get_dataset(), "Original", output_file_original, 
    prompt, model, CUB200Dataset.class_names_dict, progress_file_original
)

output_file_cropped = f"{BASE_PATH}/outputs/predictions_cropped_{timestamp}.txt"
progress_file_cropped = f"{BASE_PATH}/outputs/progress_cropped_{timestamp}.csv"
correct_cropped, total_cropped = evaluate_dataset(
    CUB200Dataset.get_dataset_cropped(), "Cropped", output_file_cropped, 
    prompt, model, CUB200Dataset.class_names_dict, progress_file_cropped
)

print(f"Original dataset accuracy: {correct_original}/{total_original} = {correct_original/total_original:.4f}")
print(f"Cropped dataset accuracy: {correct_cropped}/{total_cropped} = {correct_cropped/total_cropped:.4f}")
print(f"Results saved to: {output_file_original} and {output_file_cropped}")

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
    f.write(f"Cropped Dataset:\n")
    f.write(f"  Correct: {correct_cropped}/{total_cropped}\n")
    f.write(f"  Accuracy: {correct_cropped/total_cropped:.4f} ({correct_cropped/total_cropped*100:.2f}%)\n\n")
    f.write(f"Improvement: {(correct_cropped/total_cropped - correct_original/total_original):.4f}\n")

print(f"Summary saved to: {summary_file}")
print(f"Progress files saved to: {progress_file_original} and {progress_file_cropped}")

