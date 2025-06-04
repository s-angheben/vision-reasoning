from torchvision.datasets import INaturalist
import json

dataset = INaturalist(root="~/datasets/", download=True, version="2021_train", target_type='kingdom')

first_example = dataset[0]
print("First example:", first_example)

# print hierarchy labels of first example
cat_id = dataset.index[0][0]
dir_name = dataset.all_categories[cat_id]
pieces = dir_name.split("_")
if len(pieces) == 8:
    hierarchy = {
        "species_id": pieces[0],
        "kingdom": pieces[1],
        "phylum": pieces[2],
        "class": pieces[3],
        "order": pieces[4],
        "family": pieces[5],
        "genus": pieces[6],
        "species": pieces[7],
    }
    print("Class hierarchy:")
    for k, v in hierarchy.items():
        print(f"  {k}: {v}")
else:
    print("Could not parse class hierarchy from directory name:", dir_name)
