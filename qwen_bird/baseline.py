# Test qwen2.5VL 2b model on the Caltech-UCSD Birds 200-2011 dataset
from dataset import CUB200Dataset
from model import QwenVLModel

CUB200Dataset = CUB200Dataset(split='test')

print(CUB200Dataset.class_names_dict)

dataset = CUB200Dataset.get_dataset()
first_example = dataset[0]
print(f"First example: {first_example}")
first_example["image"].show()

dataset_cropped = CUB200Dataset.get_dataset_cropped()
first_example_cropped = dataset_cropped[0]
print(f"First example: {first_example_cropped}")
first_example_cropped["image"].show()

# Load model and make prediction
model = QwenVLModel()
prompt = f"Please identify the bird species in this image. Choose from the following list of bird species:\n\n{CUB200Dataset.prompt_class_list}\n\nProvide your answer as the species name."
prediction = model.predict(first_example["image"], prompt)

print(f"Ground truth: {CUB200Dataset.class_names_dict[first_example['label']]}")
print(f"Prediction: {prediction}")