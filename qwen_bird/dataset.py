from torch.utils.data import DataLoader
from datasets import load_dataset


class CUB200Dataset:
    def __init__(self, split='test'):
        self.CUB_200 = load_dataset("bentrevett/caltech-ucsd-birds-200-2011", split=split)
        self._cropped_dataset = None  # Cache for cropped dataset
        
        # Pre-compute class names mapping
        class_names = self.CUB_200.features["label"].names
        self.class_names_dict = {
            int(name.split(".")[0]) - 1: name.split(".")[1].strip().replace('_', ' ')
            for name in class_names
        }
        
        self.prompt_class_list = "\n".join([
            f"{i+1}. {self.class_names_dict[i]}" 
            for i in range(len(self.class_names_dict))
        ])

    def get_dataset(self):
        return self.CUB_200
    
    def get_dataset_cropped(self):
        if self._cropped_dataset is None:
            def crop_image(example):
                image = example["image"]
                bbox = example["bbox"]
                example["image"] = image.crop(bbox)
                return example
            
            self._cropped_dataset = self.CUB_200.map(
                crop_image, 
                keep_in_memory=False,
                num_proc=4  # Parallel processing for faster mapping
            )
        
        return self._cropped_dataset


