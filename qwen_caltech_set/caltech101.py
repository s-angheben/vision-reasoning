import os
import os.path
from pathlib import Path
from typing import Any, Callable, List, Optional, Union, Tuple
import requests
import zipfile
import tarfile

from PIL import Image

from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.vision import VisionDataset


class Caltech101(VisionDataset):
    """`Caltech 101 <https://data.caltech.edu/records/20086>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
            ``annotation``. Can also be a list to output a tuple with all specified
            target types.  ``category`` represents the target class, and
            ``annotation`` is a list of points from a hand-generated outline.
            Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        split (str, optional): The dataset split to use, one of: 'train', 'val', 'test'.
            If None, all samples will be included. Defaults to None.

            .. warning::

                To download the dataset `gdown <https://github.com/wkentaro/gdown>`_ is required.
    """
    
    def __init__(
        self,
        root: str,
        target_type: Union[List[str], str] = "category",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        split: Optional[str] = None,
    ) -> None:
        super().__init__(os.path.join(root, "caltech101"), transform=transform, target_transform=target_transform)
        os.makedirs(self.root, exist_ok=True)
        
        if download:
            self.download()
            
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
            
        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # Remove background class
        
        # Mapping from category to numeric index
        self.category_to_idx = {cat: i for i, cat in enumerate(self.categories)}
        
        # Load split information if split is specified
        self.split = split
        if split is not None:
            # Validate split value
            if split not in ['train', 'val', 'test']:
                raise ValueError(f"Split must be one of: 'train', 'val', 'test', got {split}")
                
            # Load split file
            split_file_path = os.path.join(os.path.dirname(__file__), 'split_coop.csv')
            if not os.path.exists(split_file_path):
                raise RuntimeError(f"Split file not found: {split_file_path}")
                
            import pandas as pd
            split_df = pd.read_csv(split_file_path)
            # Filter by the requested split
            split_files = split_df[split_df['split'] == split]['filename'].tolist()
            self.split_filenames = set(split_files)
        else:
            self.split_filenames = None
            
        # Set target type
        self.target_type = target_type if isinstance(target_type, list) else [target_type]
        self.target_type = [t for t in self.target_type if t in ["category", "annotation"]]
        if len(self.target_type) == 0:
            raise ValueError("Target type must be 'category', 'annotation' or a list containing these strings")
            
        # Load annotations
        self.annotations = []
        self.images = []
        
        # First collect all image paths
        categories_dirs = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        for category_idx, category_name in enumerate(categories_dirs):
            if category_name == "BACKGROUND_Google":
                continue
            category_dir = os.path.join(self.root, "101_ObjectCategories", category_name)
            if not os.path.isdir(category_dir):
                continue
                
            for filename in sorted(os.listdir(category_dir)):
                if not filename.endswith(".jpg"):
                    continue
                    
                # Check if the file is in the requested split
                relative_path = os.path.join(category_name, filename)
                if self.split_filenames is not None and relative_path not in self.split_filenames:
                    continue
                    
                self.images.append(os.path.join(category_dir, filename))
                self.annotations.append({"category": category_name})
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the index of the target class.
        """
        img = Image.open(self.images[index]).convert("RGB")
        
        target: Any = []
        for t in self.target_type:
            if t == "category":
                target.append(self.category_to_idx[self.annotations[index]["category"]])
            elif t == "annotation":
                # Implementation for annotation loading would go here
                pass
        
        if len(target) == 1:
            target = target[0]
            
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
        
    def __len__(self) -> int:
        return len(self.images)
        
    def _check_integrity(self) -> bool:
        # Implementation of integrity checking would go here
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))
        
    def download(self) -> None:
        # Implementation of download logic would go here
        pass


def main():
    # Create a test dataset
    dataset = Caltech101("~/datasets", download=True, split='test')
    print(f"Total samples in test set: {len(dataset)}")
    
    # Count samples per class
    class_counts = {}
    for i in range(len(dataset)):
        img, label = dataset[i]
        class_name = dataset.categories[label]
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    # Print class distribution
    print("\nClass distribution in test set:")
    print("-" * 40)
    print(f"{'Class Name':<30}Count")
    print("-" * 40)
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{class_name:<30}{count}")
    print("-" * 40)
    print(f"Total number of classes: {len(class_counts)}")
    
    # Show the first image (optional)
    dataset[0][0].show()  # Show the first image

if __name__ == "__main__":
    main()