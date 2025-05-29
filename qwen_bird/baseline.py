# Test qwen2.5VL 2b model on the Caltech-UCSD Birds 200-2011 dataset
from dataset import CUB200Dataset
from model import QwenVLModel
import os
import datetime

CUB200Dataset = CUB200Dataset(split='test')

model = QwenVLModel()
prompt = f"Please identify the bird species in this image. Choose from the following list of bird species:\n\n{CUB200Dataset.prompt_class_list}\n\nProvide your answer as the species name."

# Create output directory
os.makedirs("/home/sam/vision-reasoning/outputs", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# test original dataset
correct_original = 0
total_original = 0

output_file_original = f"/home/sam/vision-reasoning/outputs/predictions_original_{timestamp}.txt"
with open(output_file_original, "w") as f:
    f.write(f"Original Dataset Predictions - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    
    for idx, sample in enumerate(CUB200Dataset.get_dataset()):
        prediction = model.predict(sample["image"], prompt)
        ground_truth = CUB200Dataset.class_names_dict[sample['label']]
        
        # Simple accuracy check - if ground truth is in prediction
        is_correct = ground_truth.lower() in prediction.lower()
        if is_correct:
            correct_original += 1
        total_original += 1

        print(f"Sample {idx}: Ground truth: {ground_truth}")
        print(f"Sample {idx}: Prediction: {prediction}")
        print(f"Sample {idx}: Correct: {is_correct}")
        print("---")
        
        f.write(f"Sample {idx}:\n")
        f.write(f"Ground truth: {ground_truth}\n")
        f.write(f"Prediction: {prediction}\n")
        f.write(f"Correct: {is_correct}\n")
        f.write("-" * 40 + "\n\n")
    
    f.write(f"\nOriginal dataset accuracy: {correct_original}/{total_original} = {correct_original/total_original:.4f}\n")

print(f"Original dataset accuracy: {correct_original}/{total_original} = {correct_original/total_original:.4f}")

# test cropped dataset
correct_cropped = 0
total_cropped = 0

output_file_cropped = f"/home/sam/vision-reasoning/outputs/predictions_cropped_{timestamp}.txt"
with open(output_file_cropped, "w") as f:
    f.write(f"Cropped Dataset Predictions - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n\n")
    
    for idx, sample in enumerate(CUB200Dataset.get_dataset_cropped()):
        prediction = model.predict(sample["image"], prompt)
        ground_truth = CUB200Dataset.class_names_dict[sample['label']]
        
        # Simple accuracy check - if ground truth is in prediction
        is_correct = ground_truth.lower() in prediction.lower()
        if is_correct:
            correct_cropped += 1
        total_cropped += 1

        print(f"Sample {idx}: Ground truth: {ground_truth}")
        print(f"Sample {idx}: Prediction: {prediction}")
        print(f"Sample {idx}: Correct: {is_correct}")
        print("---")
        
        f.write(f"Sample {idx}:\n")
        f.write(f"Ground truth: {ground_truth}\n")
        f.write(f"Prediction: {prediction}\n")
        f.write(f"Correct: {is_correct}\n")
        f.write("-" * 40 + "\n\n")
    
    f.write(f"\nCropped dataset accuracy: {correct_cropped}/{total_cropped} = {correct_cropped/total_cropped:.4f}\n")

print(f"Cropped dataset accuracy: {correct_cropped}/{total_cropped} = {correct_cropped/total_cropped:.4f}")
print(f"Results saved to: {output_file_original} and {output_file_cropped}")

# Save summary results
summary_file = f"/home/sam/vision-reasoning/outputs/accuracy_summary_{timestamp}.txt"
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

