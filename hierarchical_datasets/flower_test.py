from flower102 import Flowers102
import json

dataset = Flowers102(root="~/datasets/", download=True)

print(json.dumps(dataset.classes, indent=2))
print(len(dataset))

i = 200
first_example = dataset[i]
label_name = dataset.classes[first_example[1]]

print("First example:", first_example)
print("label:", label_name)
print("Hierarchical label:", json.dumps(dataset.hierarchy_class[label_name], indent=2))