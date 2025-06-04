from pathlib import Path
from typing import Any, Callable, Optional, Union
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset


class Flowers102(VisionDataset):
    """`Oxford 102 Flower <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_ Dataset.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Oxford 102 Flower is an image classification dataset consisting of 102 flower categories. The
    flowers were chosen to be flowers commonly occurring in the United Kingdom. Each class consists of
    between 40 and 258 images.

    The images have large scale, pose and light variations. In addition, there are categories that
    have large variations within the category, and several very similar categories.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``"train"`` (default), ``"val"``, or ``"test"``.
        transform (callable, optional): A function/transform that takes in a PIL image or torch.Tensor, depends on the given loader,
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        loader (callable, optional): A function to load an image given its path.
            By default, it uses PIL as its image loader, but users could also pass in
            ``torchvision.io.decode_image`` for decoding image data into tensors directly.
    """

    _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    _file_dict = {  # filename, md5
        "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
    }
    _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        loader: Callable[[Union[str, Path]], Any] = default_loader,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        self._base_folder = Path(self.root) / "flowers-102"
        self._images_folder = self._base_folder / "jpg"

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        from scipy.io import loadmat

        set_ids = loadmat(self._base_folder / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()

        labels = loadmat(self._base_folder / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate((labels["labels"] - 1).tolist(), 1))

        self._labels = []
        self._image_files = []
        for image_id in image_ids:
            self._labels.append(image_id_to_label[image_id])
            self._image_files.append(self._images_folder / f"image_{image_id:05d}.jpg")

        self.loader = loader

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = self.loader(image_file)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


    def extra_repr(self) -> str:
        return f"split={self._split}"

    def _check_integrity(self):
        if not (self._images_folder.exists() and self._images_folder.is_dir()):
            return False

        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            if not check_integrity(str(self._base_folder / filename), md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            return
        download_and_extract_archive(
            f"{self._download_url_prefix}{self._file_dict['image'][0]}",
            str(self._base_folder),
            md5=self._file_dict["image"][1],
        )
        for id in ["label", "setid"]:
            filename, md5 = self._file_dict[id]
            download_url(self._download_url_prefix + filename, str(self._base_folder), md5=md5)

    classes = [
        "pink primrose",
        "hard-leaved pocket orchid",
        "canterbury bells",
        "sweet pea",
        "english marigold",
        "tiger lily",
        "moon orchid",
        "bird of paradise",
        "monkshood",
        "globe thistle",
        "snapdragon",
        "colt's foot",
        "king protea",
        "spear thistle",
        "yellow iris",
        "globe-flower",
        "purple coneflower",
        "peruvian lily",
        "balloon flower",
        "giant white arum lily",
        "fire lily",
        "pincushion flower",
        "fritillary",
        "red ginger",
        "grape hyacinth",
        "corn poppy",
        "prince of wales feathers",
        "stemless gentian",
        "artichoke",
        "sweet william",
        "carnation",
        "garden phlox",
        "love in the mist",
        "mexican aster",
        "alpine sea holly",
        "ruby-lipped cattleya",
        "cape flower",
        "great masterwort",
        "siam tulip",
        "lenten rose",
        "barbeton daisy",
        "daffodil",
        "sword lily",
        "poinsettia",
        "bolero deep blue",
        "wallflower",
        "marigold",
        "buttercup",
        "oxeye daisy",
        "common dandelion",
        "petunia",
        "wild pansy",
        "primula",
        "sunflower",
        "pelargonium",
        "bishop of llandaff",
        "gaura",
        "geranium",
        "orange dahlia",
        "pink-yellow dahlia?",
        "cautleya spicata",
        "japanese anemone",
        "black-eyed susan",
        "silverbush",
        "californian poppy",
        "osteospermum",
        "spring crocus",
        "bearded iris",
        "windflower",
        "tree poppy",
        "gazania",
        "azalea",
        "water lily",
        "rose",
        "thorn apple",
        "morning glory",
        "passion flower",
        "lotus",
        "toad lily",
        "anthurium",
        "frangipani",
        "clematis",
        "hibiscus",
        "columbine",
        "desert-rose",
        "tree mallow",
        "magnolia",
        "cyclamen",
        "watercress",
        "canna lily",
        "hippeastrum",
        "bee balm",
        "ball moss",
        "foxglove",
        "bougainvillea",
        "camellia",
        "mallow",
        "mexican petunia",
        "bromelia",
        "blanket flower",
        "trumpet creeper",
        "blackberry lily",
    ]

    # Divide classes into 4 subclasses based on botanical families and visual characteristics
    subclasses = {
        "Bulb_and_Tubular_Flowers": [
            "pink primrose", "hard-leaved pocket orchid", "tiger lily", "moon orchid", 
            "yellow iris", "peruvian lily", "fire lily", "fritillary", "grape hyacinth",
            "ruby-lipped cattleya", "siam tulip", "daffodil", "sword lily", "spring crocus",
            "bearded iris", "water lily", "lotus", "toad lily", "anthurium", "canna lily",
            "hippeastrum", "blackberry lily"
        ],
        "Composite_and_Daisy_Flowers": [
            "globe thistle", "spear thistle", "purple coneflower", "mexican aster",
            "barbeton daisy", "oxeye daisy", "common dandelion", "sunflower", 
            "black-eyed susan", "osteospermum", "gazania", "orange dahlia", 
            "pink-yellow dahlia?", "blanket flower"
        ],
        "Simple_and_Bell_Flowers": [
            "canterbury bells", "sweet pea", "english marigold", "bird of paradise",
            "monkshood", "snapdragon", "colt's foot", "balloon flower", "giant white arum lily",
            "pincushion flower", "corn poppy", "prince of wales feathers", "stemless gentian",
            "sweet william", "carnation", "garden phlox", "love in the mist", "alpine sea holly",
            "cape flower", "great masterwort", "lenten rose", "poinsettia", "bolero deep blue",
            "wallflower", "marigold", "buttercup", "petunia", "wild pansy", "primula",
            "pelargonium", "bishop of llandaff", "gaura", "geranium", "californian poppy",
            "windflower", "tree poppy", "morning glory", "passion flower", "columbine",
            "cyclamen", "bee balm", "foxglove", "mexican petunia"
        ],
        "Shrub_and_Tree_Flowers": [
            "king protea", "globe-flower", "red ginger", "artichoke", "cautleya spicata",
            "japanese anemone", "silverbush", "azalea", "rose", "thorn apple", "frangipani",
            "clematis", "hibiscus", "desert-rose", "tree mallow", "magnolia", "watercress",
            "ball moss", "bougainvillea", "camellia", "mallow", "bromelia", "trumpet creeper"
        ]
    }

    hierarchy_class = {
        # Bulb_and_Tubular_Flowers
        "pink primrose": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Primrose",
                "synonyms": ["Primula", "Cowslip (type)", "Oxlip (type)", "Polyanthus (type)"]
            },
            "Level 4": {
                "label": "Pink Primrose",
                "synonyms": ["Primula rosea", "Himalayan Meadow Primrose", "Rosy Primrose"]
            }
        },
        "hard-leaved pocket orchid": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Orchid",
                "synonyms": ["Orchidaceae", "Orchid Flower", "Orchid Plant", "Orchid Blossom"]
            },
            "Level 4": {
                "label": "Hard-leaved Pocket Orchid",
                "synonyms": ["Paphiopedilum micranthum", "Silver Slipper Orchid", "Pocket-leaf Orchid"]
            }
        },
        "tiger lily": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Lily",
                "synonyms": ["Lilium", "True Lily", "Lily Flower", "Lily Plant"]
            },
            "Level 4": {
                "label": "Tiger Lily",
                "synonyms": ["Lilium lancifolium", "Lilium tigrinum", "Orange Tiger Lily", "Spotted Lily"]
            }
        },
        "moon orchid": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Orchid",
                "synonyms": ["Orchidaceae", "Orchid Flower", "Orchid Plant", "Orchid Blossom"]
            },
            "Level 4": {
                "label": "Moon Orchid",
                "synonyms": ["Phalaenopsis amabilis", "Moth Orchid", "White Moon Orchid"]
            }
        },
        "yellow iris": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Iris",
                "synonyms": ["Iris Flower", "Iris Plant", "Flag Iris", "Sword Lily (iris type)"]
            },
            "Level 4": {
                "label": "Yellow Iris",
                "synonyms": ["Iris pseudacorus", "Yellow Flag", "Water Flag"]
            }
        },
        "peruvian lily": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Alstroemeria",
                "synonyms": ["Peruvian Lily", "Lily of the Incas", "Alstroemeria Flower"]
            },
            "Level 4": {
                "label": "Peruvian Lily",
                "synonyms": ["Alstroemeria aurea", "Golden Peruvian Lily", "Alstroemeria"]
            }
        },
        "fire lily": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Lily",
                "synonyms": ["Lilium", "True Lily", "Lily Flower", "Lily Plant"]
            },
            "Level 4": {
                "label": "Fire Lily",
                "synonyms": ["Lilium bulbiferum", "Orange Lily", "Fire-colored Lily"]
            }
        },
        "fritillary": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Fritillary",
                "synonyms": ["Fritillaria", "Fritillary Flower", "Checkered Lily", "Snake's Head"]
            },
            "Level 4": {
                "label": "Fritillary",
                "synonyms": ["Fritillaria meleagris", "Snake's Head Fritillary", "Chess Flower"]
            }
        },
        "grape hyacinth": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Hyacinth",
                "synonyms": ["Hyacinth Flower", "Hyacinth Plant", "Muscari", "Grape Hyacinth"]
            },
            "Level 4": {
                "label": "Grape Hyacinth",
                "synonyms": ["Muscari armeniacum", "Blue Grape Hyacinth", "Cluster Hyacinth"]
            }
        },
        "ruby-lipped cattleya": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Orchid",
                "synonyms": ["Orchidaceae", "Orchid Flower", "Orchid Plant", "Orchid Blossom"]
            },
            "Level 4": {
                "label": "Ruby-lipped Cattleya",
                "synonyms": ["Cattleya labiata", "Crimson-lipped Cattleya", "Corsage Orchid"]
            }
        },
        "siam tulip": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Curcuma",
                "synonyms": ["Siam Tulip", "Hidden Lily", "Summer Tulip", "Curcuma Flower"]
            },
            "Level 4": {
                "label": "Siam Tulip",
                "synonyms": ["Curcuma alismatifolia", "Thai Tulip", "Patumma"]
            }
        },
        "daffodil": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Daffodil",
                "synonyms": ["Narcissus", "Daffodil Flower", "Jonquil", "Lent Lily"]
            },
            "Level 4": {
                "label": "Daffodil",
                "synonyms": ["Narcissus pseudonarcissus", "Wild Daffodil", "Trumpet Daffodil"]
            }
        },
        "sword lily": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Gladiolus",
                "synonyms": ["Sword Lily", "Gladiolus Flower", "Gladiolus Plant"]
            },
            "Level 4": {
                "label": "Sword Lily",
                "synonyms": ["Gladiolus hortulanus", "Garden Gladiolus", "Common Sword Lily"]
            }
        },
        "spring crocus": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Crocus",
                "synonyms": ["Crocus Flower", "Crocus Plant", "Spring Crocus"]
            },
            "Level 4": {
                "label": "Spring Crocus",
                "synonyms": ["Crocus vernus", "Giant Crocus", "Dutch Crocus"]
            }
        },
        "bearded iris": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Iris",
                "synonyms": ["Iris Flower", "Iris Plant", "Flag Iris", "Sword Lily (iris type)"]
            },
            "Level 4": {
                "label": "Bearded Iris",
                "synonyms": ["Iris germanica", "German Iris", "Tall Bearded Iris"]
            }
        },
        "water lily": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Water Lily",
                "synonyms": ["Nymphaea", "Aquatic Lily", "Pond Lily", "Waterlily"]
            },
            "Level 4": {
                "label": "Water Lily",
                "synonyms": ["Nymphaea alba", "European White Water Lily", "White Waterlily"]
            }
        },
        "lotus": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Lotus",
                "synonyms": ["Nelumbo", "Sacred Lotus", "Water Lotus", "Lotus Flower"]
            },
            "Level 4": {
                "label": "Lotus",
                "synonyms": ["Nelumbo nucifera", "Indian Lotus", "Sacred Lotus"]
            }
        },
        "toad lily": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Toad Lily",
                "synonyms": ["Tricyrtis", "Orchid-like Lily", "Shade Lily"]
            },
            "Level 4": {
                "label": "Toad Lily",
                "synonyms": ["Tricyrtis hirta", "Hairy Toad Lily", "Japanese Toad Lily"]
            }
        },
        "anthurium": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Anthurium",
                "synonyms": ["Flamingo Flower", "Laceleaf", "Tailflower", "Painter's Palette"]
            },
            "Level 4": {
                "label": "Anthurium",
                "synonyms": ["Anthurium andraeanum", "Painter's Anthurium", "Red Flamingo Flower"]
            }
        },
        "canna lily": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Canna",
                "synonyms": ["Canna Lily", "Indian Shot", "Canna Flower"]
            },
            "Level 4": {
                "label": "Canna Lily",
                "synonyms": ["Canna indica", "Indian Canna", "Edible Canna"]
            }
        },
        "hippeastrum": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Hippeastrum",
                "synonyms": ["Amaryllis", "Hippeastrum Flower", "Belladonna Lily (hippeastrum type)"]
            },
            "Level 4": {
                "label": "Hippeastrum",
                "synonyms": ["Hippeastrum hybridum", "Dutch Amaryllis", "Christmas Amaryllis"]
            }
        },
        "blackberry lily": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Iris",
                "synonyms": ["Iris Flower", "Iris Plant", "Flag Iris", "Sword Lily (iris type)"]
            },
            "Level 4": {
                "label": "Blackberry Lily",
                "synonyms": ["Iris domestica", "Leopard Lily", "Belamcanda chinensis"]
            }
        },

        # Composite_and_Daisy_Flowers
        "globe thistle": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Thistle",
                "synonyms": ["Asteraceae Thistle", "Spiny Flower", "Thistle Plant"]
            },
            "Level 4": {
                "label": "Globe Thistle",
                "synonyms": ["Echinops", "Echinops ritro", "Blue Globe Thistle"]
            }
        },
        "spear thistle": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Thistle",
                "synonyms": ["Asteraceae Thistle", "Spiny Flower", "Thistle Plant"]
            },
            "Level 4": {
                "label": "Spear Thistle",
                "synonyms": ["Cirsium vulgare", "Bull Thistle", "Common Thistle"]
            }
        },
        "purple coneflower": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Coneflower",
                "synonyms": ["Echinacea", "Hedgehog Flower", "Daisy-like Flower", "Prairie Flower (coneflower type)"]
            },
            "Level 4": {
                "label": "Purple Coneflower",
                "synonyms": ["Echinacea purpurea", "Eastern Purple Coneflower", "Purple Rudbeckia", "Hedgehog Coneflower (purple)"]
            }
        },
        "mexican aster": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Aster",
                "synonyms": ["Asteraceae", "Aster Flower", "Daisy-like Aster"]
            },
            "Level 4": {
                "label": "Mexican Aster",
                "synonyms": ["Cosmos bipinnatus", "Garden Cosmos", "Cosmos Flower"]
            }
        },
        "barbeton daisy": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Daisy",
                "synonyms": ["Asteraceae Daisy", "Daisy Flower", "Gerbera"]
            },
            "Level 4": {
                "label": "Barbeton Daisy",
                "synonyms": ["Gerbera jamesonii", "Transvaal Daisy", "African Daisy"]
            }
        },
        "oxeye daisy": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Daisy",
                "synonyms": ["Asteraceae Daisy", "Daisy Flower", "Leucanthemum"]
            },
            "Level 4": {
                "label": "Oxeye Daisy",
                "synonyms": ["Leucanthemum vulgare", "Moon Daisy", "Dog Daisy"]
            }
        },
        "common dandelion": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Dandelion",
                "synonyms": ["Taraxacum", "Dandelion Flower", "Blowball"]
            },
            "Level 4": {
                "label": "Common Dandelion",
                "synonyms": ["Taraxacum officinale", "Lion's Tooth", "Wild Dandelion"]
            }
        },
        "sunflower": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Sunflower",
                "synonyms": ["Helianthus", "Sunflower Plant", "Sunflower Blossom"]
            },
            "Level 4": {
                "label": "Sunflower",
                "synonyms": ["Helianthus annuus", "Common Sunflower", "Giant Sunflower"]
            }
        },
        "black-eyed susan": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Rudbeckia",
                "synonyms": ["Black-eyed Susan", "Coneflower (rudbeckia type)", "Rudbeckia Flower"]
            },
            "Level 4": {
                "label": "Black-eyed Susan",
                "synonyms": ["Rudbeckia hirta", "Brown-eyed Susan", "Gloriosa Daisy"]
            }
        },
        "osteospermum": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Daisy",
                "synonyms": ["Asteraceae Daisy", "Daisy Flower", "African Daisy"]
            },
            "Level 4": {
                "label": "Osteospermum",
                "synonyms": ["Osteospermum ecklonis", "Cape Daisy", "Blue-eyed Daisy"]
            }
        },
        "gazania": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Gazania",
                "synonyms": ["Treasure Flower", "Gazania Flower", "African Daisy (gazania type)"]
            },
            "Level 4": {
                "label": "Gazania",
                "synonyms": ["Gazania rigens", "Trailing Gazania", "Coastal Gazania"]
            }
        },
        "orange dahlia": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Dahlia",
                "synonyms": ["Dahlia Flower", "Dahlia Plant", "Asteraceae Dahlia"]
            },
            "Level 4": {
                "label": "Orange Dahlia",
                "synonyms": ["Dahlia pinnata (orange)", "Orange-flowered Dahlia"]
            }
        },
        "pink-yellow dahlia?": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Dahlia",
                "synonyms": ["Dahlia Flower", "Dahlia Plant", "Asteraceae Dahlia"]
            },
            "Level 4": {
                "label": "Pink-yellow Dahlia",
                "synonyms": ["Dahlia hybrid (pink-yellow)", "Bicolor Dahlia"]
            }
        },
        "blanket flower": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Gaillardia",
                "synonyms": ["Blanket Flower", "Gaillardia Flower", "Indian Blanket"]
            },
            "Level 4": {
                "label": "Blanket Flower",
                "synonyms": ["Gaillardia pulchella", "Firewheel", "Common Blanket Flower"]
            }
        },

        # Simple_and_Bell_Flowers
        "canterbury bells": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Bellflower",
                "synonyms": ["Campanula", "Bell-shaped Flower", "Campanulaceae Flower"]
            },
            "Level 4": {
                "label": "Canterbury Bells",
                "synonyms": ["Campanula medium", "Cup-and-saucer", "Bells of Canterbury"]
            }
        },
        "sweet pea": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Pea Flower",
                "synonyms": ["Lathyrus", "Fabaceae Flower", "Legume Flower"]
            },
            "Level 4": {
                "label": "Sweet Pea",
                "synonyms": ["Lathyrus odoratus", "Fragrant Pea", "Garden Sweet Pea"]
            }
        },
        "english marigold": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Marigold",
                "synonyms": ["Calendula", "Pot Marigold", "Asteraceae Marigold"]
            },
            "Level 4": {
                "label": "English Marigold",
                "synonyms": ["Calendula officinalis", "Common Marigold", "Scotch Marigold"]
            }
        },
        "bird of paradise": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Strelitzia",
                "synonyms": ["Crane Flower", "Bird Flower", "Strelitziaceae Flower"]
            },
            "Level 4": {
                "label": "Bird of Paradise",
                "synonyms": ["Strelitzia reginae", "Orange Bird of Paradise", "Crane Flower"]
            }
        },
        "monkshood": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Aconite",
                "synonyms": ["Monkshood Flower", "Wolfsbane", "Helmet Flower"]
            },
            "Level 4": {
                "label": "Monkshood",
                "synonyms": ["Aconitum napellus", "Blue Monkshood", "Common Monkshood"]
            }
        },
        "snapdragon": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Snapdragon",
                "synonyms": ["Antirrhinum", "Dragon Flower", "Toadflax"]
            },
            "Level 4": {
                "label": "Snapdragon",
                "synonyms": ["Antirrhinum majus", "Garden Snapdragon", "Common Snapdragon"]
            }
        },
        "colt's foot": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Coltsfoot",
                "synonyms": ["Tussilago", "Yellow Daisy", "Asteraceae Coltsfoot"]
            },
            "Level 4": {
                "label": "Colt's Foot",
                "synonyms": ["Tussilago farfara", "Son-before-father", "Horsehoof"]
            }
        },
        "balloon flower": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Balloon Flower",
                "synonyms": ["Platycodon", "Bellflower (platycodon type)", "Campanulaceae Balloon Flower"]
            },
            "Level 4": {
                "label": "Balloon Flower",
                "synonyms": ["Platycodon grandiflorus", "Chinese Bellflower", "Japanese Balloon Flower"]
            }
        },
        "giant white arum lily": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Arum Lily",
                "synonyms": ["Zantedeschia", "Calla Lily", "White Lily"]
            },
            "Level 4": {
                "label": "Giant White Arum Lily",
                "synonyms": ["Zantedeschia aethiopica", "White Calla Lily", "Common Arum Lily"]
            }
        },
        "pincushion flower": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Pincushion Flower",
                "synonyms": ["Scabiosa", "Dipsacaceae Flower", "Scabious"]
            },
            "Level 4": {
                "label": "Pincushion Flower",
                "synonyms": ["Scabiosa atropurpurea", "Sweet Scabious", "Mournful Widow"]
            }
        },
        "corn poppy": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Poppy",
                "synonyms": ["Papaver", "Papaveraceae Flower", "Red Poppy"]
            },
            "Level 4": {
                "label": "Corn Poppy",
                "synonyms": ["Papaver rhoeas", "Field Poppy", "Flanders Poppy"]
            }
        },
        "prince of wales feathers": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Amaranth",
                "synonyms": ["Celosia", "Feather Flower", "Plumed Amaranth"]
            },
            "Level 4": {
                "label": "Prince of Wales Feathers",
                "synonyms": ["Celosia argentea", "Silver Cockscomb", "Plumed Celosia"]
            }
        },
        "stemless gentian": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Gentian",
                "synonyms": ["Gentiana", "Gentianaceae Flower", "Blue Gentian"]
            },
            "Level 4": {
                "label": "Stemless Gentian",
                "synonyms": ["Gentiana acaulis", "Trumpet Gentian", "Stemless Blue Gentian"]
            }
        },
        "sweet william": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Pink Flower",
                "synonyms": ["Dianthus", "Caryophyllaceae Flower", "Clove Pink", "Garden Pink"]
            },
            "Level 4": {
                "label": "Sweet William",
                "synonyms": ["Dianthus barbatus", "Bunch Pink", "Wild Sweet William", "Bearded Pink"]
            }
        },
        "carnation": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Pink Flower",
                "synonyms": ["Dianthus", "Caryophyllaceae Flower", "Clove Pink", "Garden Pink"]
            },
            "Level 4": {
                "label": "Carnation",
                "synonyms": ["Dianthus caryophyllus", "Clove Pink", "Grenadine"]
            }
        },
        "garden phlox": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Phlox",
                "synonyms": ["Polemoniaceae Flower", "Phlox Flower", "Phlox Plant"]
            },
            "Level 4": {
                "label": "Garden Phlox",
                "synonyms": ["Phlox paniculata", "Summer Phlox", "Perennial Phlox"]
            }
        },
        "love in the mist": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Nigella",
                "synonyms": ["Love-in-a-mist Flower", "Ranunculaceae Nigella", "Devil-in-the-bush"]
            },
            "Level 4": {
                "label": "Love in the Mist",
                "synonyms": ["Nigella damascena", "Devil in the Bush", "Ragged Lady"]
            }
        },
        "alpine sea holly": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Sea Holly",
                "synonyms": ["Eryngium", "Apiaceae Flower", "Spiny Blue Flower"]
            },
            "Level 4": {
                "label": "Alpine Sea Holly",
                "synonyms": ["Eryngium alpinum", "Blue Sea Holly", "Alpine Eryngium"]
            }
        },
        "cape flower": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Cape Flower",
                "synonyms": ["South African Flower", "Cape Floral", "Cape Plant"]
            },
            "Level 4": {
                "label": "Cape Flower",
                "synonyms": [None]
            }
        },
        "great masterwort": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Masterwort",
                "synonyms": ["Astrantia", "Apiaceae Flower", "Star Flower"]
            },
            "Level 4": {
                "label": "Great Masterwort",
                "synonyms": ["Astrantia major", "Large Masterwort", "Great Astrantia"]
            }
        },
        "lenten rose": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Hellebore",
                "synonyms": ["Helleborus", "Ranunculaceae Hellebore", "Winter Rose"]
            },
            "Level 4": {
                "label": "Lenten Rose",
                "synonyms": ["Helleborus orientalis", "Spring Hellebore", "Oriental Hellebore"]
            }
        },
        "poinsettia": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Poinsettia",
                "synonyms": ["Euphorbia pulcherrima", "Christmas Flower", "Spurge"]
            },
            "Level 4": {
                "label": "Poinsettia",
                "synonyms": ["Euphorbia pulcherrima", "Christmas Star", "Mexican Flame Leaf"]
            }
        },
        "bolero deep blue": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Blue Flower",
                "synonyms": ["Blue Ornamental Flower", "Blue Garden Flower", "Blue Blossom"]
            },
            "Level 4": {
                "label": "Bolero Deep Blue",
                "synonyms": [None]
            }
        },
        "wallflower": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Wallflower",
                "synonyms": ["Erysimum", "Brassicaceae Flower", "Cheiranthus"]
            },
            "Level 4": {
                "label": "Wallflower",
                "synonyms": ["Erysimum cheiri", "Common Wallflower", "Golden Wallflower"]
            }
        },
        "marigold": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Marigold",
                "synonyms": ["Tagetes", "Asteraceae Marigold", "African Marigold"]
            },
            "Level 4": {
                "label": "Marigold",
                "synonyms": ["Tagetes erecta", "Mexican Marigold", "Aztec Marigold"]
            }
        },
        "buttercup": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Buttercup",
                "synonyms": ["Ranunculus", "Ranunculaceae Flower", "Yellow Buttercup"]
            },
            "Level 4": {
                "label": "Buttercup",
                "synonyms": ["Ranunculus acris", "Meadow Buttercup", "Tall Buttercup"]
            }
        },
        "petunia": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Petunia",
                "synonyms": ["Solanaceae Flower", "Petunia Flower", "Garden Petunia"]
            },
            "Level 4": {
                "label": "Petunia",
                "synonyms": ["Petunia × atkinsiana", "Hybrid Petunia", "Grandiflora Petunia"]
            }
        },
        "wild pansy": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Pansy",
                "synonyms": ["Viola", "Violet Flower", "Heartsease"]
            },
            "Level 4": {
                "label": "Wild Pansy",
                "synonyms": ["Viola tricolor", "Heart's Ease", "Johnny Jump Up"]
            }
        },
        "primula": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Primrose",
                "synonyms": ["Primula", "Primulaceae Flower", "Primrose Flower"]
            },
            "Level 4": {
                "label": "Primula",
                "synonyms": [None]
            }
        },
        "pelargonium": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Geranium",
                "synonyms": ["Pelargonium", "Geraniaceae Flower", "Storksbill"]
            },
            "Level 4": {
                "label": "Pelargonium",
                "synonyms": ["Pelargonium hortorum", "Zonal Geranium", "Garden Geranium"]
            }
        },
        "bishop of llandaff": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Dahlia",
                "synonyms": ["Dahlia Flower", "Asteraceae Dahlia", "Garden Dahlia"]
            },
            "Level 4": {
                "label": "Bishop of Llandaff",
                "synonyms": ["Dahlia 'Bishop of Llandaff'", "Red Dahlia", "Dark-leaved Dahlia"]
            }
        },
        "gaura": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Gaura",
                "synonyms": ["Oenothera", "Evening Primrose Family", "Bee Blossom"]
            },
            "Level 4": {
                "label": "Gaura",
                "synonyms": ["Oenothera lindheimeri", "Lindheimer's Beeblossom", "White Gaura"]
            }
        },
        "geranium": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Geranium",
                "synonyms": ["Cranesbill", "Geraniaceae Flower", "Hardy Geranium"]
            },
            "Level 4": {
                "label": "Geranium",
                "synonyms": ["Geranium pratense", "Meadow Cranesbill", "Wild Geranium"]
            }
        },
        "californian poppy": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Poppy",
                "synonyms": ["Papaveraceae Flower", "Eschscholzia", "California Poppy"]
            },
            "Level 4": {
                "label": "Californian Poppy",
                "synonyms": ["Eschscholzia californica", "Golden Poppy", "California Sunlight"]
            }
        },
        "windflower": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Anemone",
                "synonyms": ["Windflower", "Ranunculaceae Anemone", "Anemone Flower"]
            },
            "Level 4": {
                "label": "Windflower",
                "synonyms": ["Anemone nemorosa", "Wood Anemone", "European Windflower"]
            }
        },
        "tree poppy": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Tree Poppy",
                "synonyms": ["Dendromecon", "Papaveraceae Tree Poppy", "Shrubby Poppy"]
            },
            "Level 4": {
                "label": "Tree Poppy",
                "synonyms": ["Dendromecon rigida", "Bush Poppy", "Yellow Tree Poppy"]
            }
        },
        "morning glory": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Morning Glory",
                "synonyms": ["Ipomoea", "Convolvulaceae Flower", "Bindweed"]
            },
            "Level 4": {
                "label": "Morning Glory",
                "synonyms": ["Ipomoea purpurea", "Common Morning Glory", "Purple Morning Glory"]
            }
        },
        "passion flower": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Passion Flower",
                "synonyms": ["Passiflora", "Passion Vine", "Passifloraceae Flower"]
            },
            "Level 4": {
                "label": "Passion Flower",
                "synonyms": ["Passiflora caerulea", "Blue Passionflower", "Common Passion Flower"]
            }
        },
        "columbine": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Columbine",
                "synonyms": ["Aquilegia", "Ranunculaceae Columbine", "Granny's Bonnet"]
            },
            "Level 4": {
                "label": "Columbine",
                "synonyms": ["Aquilegia vulgaris", "European Columbine", "Common Columbine"]
            }
        },
        "cyclamen": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Cyclamen",
                "synonyms": ["Primulaceae Cyclamen", "Cyclamen Flower", "Sowbread"]
            },
            "Level 4": {
                "label": "Cyclamen",
                "synonyms": ["Cyclamen persicum", "Florist's Cyclamen", "Persian Cyclamen"]
            }
        },
        "bee balm": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Bee Balm",
                "synonyms": ["Monarda", "Lamiaceae Flower", "Bergamot"]
            },
            "Level 4": {
                "label": "Bee Balm",
                "synonyms": ["Monarda didyma", "Oswego Tea", "Scarlet Beebalm"]
            }
        },
        "foxglove": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Foxglove",
                "synonyms": ["Digitalis", "Plantaginaceae Flower", "Bell-shaped Foxglove"]
            },
            "Level 4": {
                "label": "Foxglove",
                "synonyms": ["Digitalis purpurea", "Common Foxglove", "Purple Foxglove"]
            }
        },
        "mexican petunia": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Petunia",
                "synonyms": ["Ruellia", "Acanthaceae Flower", "Wild Petunia"]
            },
            "Level 4": {
                "label": "Mexican Petunia",
                "synonyms": ["Ruellia simplex", "Purple Showers", "Mexican Bluebell"]
            }
        },

        "king protea": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Protea",
                "synonyms": ["Proteaceae Flower", "Sugarbush", "Protea Flower"]
            },
            "Level 4": {
                "label": "King Protea",
                "synonyms": ["Protea cynaroides", "Giant Protea", "King Sugarbush"]
            }
        },
        "globe-flower": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Globe Flower",
                "synonyms": ["Trollius", "Buttercup Family Globe Flower", "Round Flower"]
            },
            "Level 4": {
                "label": "Globe-flower",
                "synonyms": ["Trollius europaeus", "European Globe Flower", "Yellow Globe Flower"]
            }
        },
        "red ginger": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Ginger Flower",
                "synonyms": ["Zingiberaceae Flower", "Ginger Plant Flower", "Tropical Ginger"]
            },
            "Level 4": {
                "label": "Red Ginger",
                "synonyms": ["Alpinia purpurata", "Ostrich Plume", "Pink Cone Ginger"]
            }
        },
        "artichoke": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Thistle Flower",
                "synonyms": ["Asteraceae Thistle", "Thistle Plant", "Spiny Flower"]
            },
            "Level 4": {
                "label": "Artichoke",
                "synonyms": ["Cynara cardunculus", "Globe Artichoke", "Edible Artichoke"]
            }
        },
        "cautleya spicata": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Ginger Flower",
                "synonyms": ["Zingiberaceae Flower", "Ginger Plant Flower", "Tropical Ginger"]
            },
            "Level 4": {
                "label": "Cautleya spicata",
                "synonyms": ["Spiked Ginger Lily", "Cautleya", "Spicate Cautleya"]
            }
        },
        "japanese anemone": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Anemone",
                "synonyms": ["Windflower", "Ranunculaceae Anemone", "Anemone Flower"]
            },
            "Level 4": {
                "label": "Japanese Anemone",
                "synonyms": ["Anemone hupehensis", "Japanese Windflower", "Chinese Anemone"]
            }
        },
        "silverbush": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Convolvulus",
                "synonyms": ["Bindweed", "Morning Glory Family", "Convolvulaceae Flower"]
            },
            "Level 4": {
                "label": "Silverbush",
                "synonyms": ["Convolvulus cneorum", "Shrubby Bindweed", "Silver Bush"]
            }
        },
        "azalea": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Rhododendron",
                "synonyms": ["Azalea", "Ericaceae Flower", "Rhododendron Flower"]
            },
            "Level 4": {
                "label": "Azalea",
                "synonyms": [None]
            }
        },
        "rose": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Rose",
                "synonyms": ["Rosaceae Flower", "Rose Flower", "Garden Rose"]
            },
            "Level 4": {
                "label": "Rose",
                "synonyms": [None]
            }
        },
        "thorn apple": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Datura",
                "synonyms": ["Thorn Apple", "Nightshade Family", "Solanaceae Flower"]
            },
            "Level 4": {
                "label": "Thorn Apple",
                "synonyms": ["Datura stramonium", "Jimsonweed", "Devil's Snare"]
            }
        },
        "frangipani": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Plumeria",
                "synonyms": ["Frangipani", "Dogbane Family", "Plumeria Flower"]
            },
            "Level 4": {
                "label": "Frangipani",
                "synonyms": [None]
            }
        },
        "clematis": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Clematis",
                "synonyms": ["Buttercup Family Clematis", "Ranunculaceae Clematis", "Clematis Flower"]
            },
            "Level 4": {
                "label": "Clematis",
                "synonyms": [None]
            }
        },
        "hibiscus": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Hibiscus",
                "synonyms": ["Mallow Family Hibiscus", "Malvaceae Hibiscus", "Hibiscus Flower"]
            },
            "Level 4": {
                "label": "Hibiscus",
                "synonyms": [None]
            }
        },
        "desert-rose": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Adenium",
                "synonyms": ["Desert Rose", "Dogbane Family Adenium", "Adenium Flower"]
            },
            "Level 4": {
                "label": "Desert-rose",
                "synonyms": ["Adenium obesum", "Mock Azalea", "Impala Lily"]
            }
        },
        "tree mallow": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Mallow",
                "synonyms": ["Malvaceae Mallow", "Mallow Flower", "Shrubby Mallow"]
            },
            "Level 4": {
                "label": "Tree Mallow",
                "synonyms": ["Malva arborea", "Lavatera arborea", "Shrubby Mallow"]
            }
        },
        "magnolia": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Magnolia",
                "synonyms": ["Magnoliaceae Magnolia", "Magnolia Flower", "Magnolia Tree"]
            },
            "Level 4": {
                "label": "Magnolia",
                "synonyms": [None]
            }
        },
        "watercress": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Nasturtium",
                "synonyms": ["Watercress", "Brassicaceae Nasturtium", "Aquatic Cress"]
            },
            "Level 4": {
                "label": "Watercress",
                "synonyms": ["Nasturtium officinale", "True Watercress", "Aquatic Nasturtium"]
            }
        },
        "ball moss": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Tillandsia",
                "synonyms": ["Ball Moss", "Bromeliad", "Air Plant"]
            },
            "Level 4": {
                "label": "Ball Moss",
                "synonyms": ["Tillandsia recurvata", "Ball Tillandsia", "Epiphytic Ball Moss"]
            }
        },
        "bougainvillea": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Bougainvillea",
                "synonyms": ["Nyctaginaceae Bougainvillea", "Bougainvillea Flower", "Paper Flower"]
            },
            "Level 4": {
                "label": "Bougainvillea",
                "synonyms": [None]
            }
        },
        "camellia": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Camellia",
                "synonyms": ["Theaceae Camellia", "Camellia Flower", "Tea Flower"]
            },
            "Level 4": {
                "label": "Camellia",
                "synonyms": [None]
            }
        },
        "mallow": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Mallow",
                "synonyms": ["Malvaceae Mallow", "Mallow Flower", "Common Mallow"]
            },
            "Level 4": {
                "label": "Mallow",
                "synonyms": [None]
            }
        },
        "bromelia": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Bromeliad",
                "synonyms": ["Bromeliaceae Flower", "Bromelia", "Pineapple Family Flower"]
            },
            "Level 4": {
                "label": "Bromelia",
                "synonyms": [None]
            }
        },
        "trumpet creeper": {
            "Level 1": {
                "label": "Plant",
                "synonyms": ["Vegetation", "Flora", "Greenery", "Botanical Life"]
            },
            "Level 2": {
                "label": "Flowering Plant",
                "synonyms": ["Angiosperm", "Bloom-bearing Plant", "Floral Plant", "Flower Producer", "Ornamental Plant", "Flower", "Blossom"]
            },
            "Level 3": {
                "label": "Campsis",
                "synonyms": ["Trumpet Creeper", "Bignoniaceae Flower", "Trumpet Vine"]
            },
            "Level 4": {
                "label": "Trumpet Creeper",
                "synonyms": ["Campsis radicans", "Cow Itch Vine", "Trumpet Vine"]
            }
        }

    }