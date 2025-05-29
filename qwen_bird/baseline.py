# Test qwen2.5VL 2b model on the Caltech-UCSD Birds 200-2011 dataset
from datasets import load_dataset

CUB_200_test = load_dataset("bentrevett/caltech-ucsd-birds-200-2011", split="test")

class_names = CUB_200_test.features["label"].names
class_names_dict = {}
for class_name in class_names:
    id = class_name.split(".")[0]
    class_names_dict[id] = class_name.split(".")[1].strip()

# print(class_names_dict)
print(class_names_dict)


first_example = CUB_200_test[0]
print(f"First example: {first_example}")

import torch
import torchvision
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# Create a formatted list of classes for the prompt
class_list = "\n".join([f"{i+1}. {class_names_dict[str(i+1).zfill(3)]}" for i in range(len(class_names_dict))])
print(class_list)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": first_example["image"],
            },
            {"type": "text", "text": f"Please identify the bird species in this image. Choose from the following list of bird species:\n\n{class_list}\n\nProvide your answer as the species name."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# Write output to file
import os
import datetime

# Create output directory if it doesn't exist
os.makedirs("/home/samuele.angheben/vision-reasoning/outputs", exist_ok=True)

# Generate timestamp for filename
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"/home/samuele.angheben/vision-reasoning/outputs/baseline_output_{timestamp}.txt"

with open(output_file, "w") as f:
    f.write(f"Bird Classification Results\n")
    f.write(f"==========================\n\n")
    f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: Qwen2.5-VL-3B-Instruct\n")
    f.write(f"Dataset: Caltech-UCSD Birds 200-2011 (test set)\n")
    f.write(f"Test example index: 43\n\n")
    f.write(f"Ground truth label: {first_example['label']}\n")
    f.write(f"Ground truth class: {class_names_dict[str(first_example['label'] + 1).zfill(3)]}\n\n")
    f.write(f"Model prediction:\n{output_text[0]}\n")

print(f"Output written to: {output_file}")
print(f"Model prediction: {output_text[0]}")