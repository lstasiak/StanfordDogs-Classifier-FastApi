import os
from typing import Dict, List, Tuple, Union

from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

from project_utils import get_labels
from settings import DATA_DIR_STRUCT, DATA_SPLIT


class DatasetCollector:
    """
    Class representation of the object for loading datasets into dataloaders.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        batch_size: int,
        data_root: Union[str, None] = None,
        organize=False,
        split_ratio=None,
    ) -> None:
        """
        :param organize: set as True when data directory is not separated
         into [train, test, ...] sub-dirs
        :param split_ratio: list of split values according
         to which data will be split
        :param img_size: image size in the format: (width, height)
        :param batch_size: batch size
        :param data_root: data directory path
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.data_root = data_root
        self.transforms = ImageTransformer.get_image_transforms(img_size)
        self.organize = organize
        self.split_ratio = split_ratio
        self.datasets = self.create_datasets()
        self._classes = self.datasets.get("train", "val").classes

    def create_datasets(self) -> Dict[str, datasets.ImageFolder]:
        """
        Return Dict with datasets according to data split defined in DATA_SPLIT setting.
        """
        if self.organize and self.data_root is not None:
            classes2labels = get_labels(self.data_root)
            dataset = StanfordDogsImageDataset(
                self.data_root, classes_to_labels=classes2labels
            )
            subsets: List[Subset] = random_split(dataset, self.split_ratio)

            return {
                phase: DatasetFromSubset(
                    subset,
                    self.data_root,
                    transform=self.transforms[phase],
                    classes_to_labels=classes2labels,
                )
                for phase, subset in zip(DATA_SPLIT, subsets)
            }
        else:
            return {
                phase: datasets.ImageFolder(path, transform=self.transforms[phase])
                for phase, path in DATA_DIR_STRUCT.items()
            }

    def get_dataloaders(
        self, shuffle: bool = True, num_workers: int = 0
    ) -> Dict[str, DataLoader]:
        return {
            phase: DataLoader(
                self.datasets[phase],
                self.batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
            )
            for phase in DATA_SPLIT
        }

    def get_classes(self) -> List[str]:
        return self._classes

    def get_dataset_summary(self) -> str:
        classes = self.get_classes()
        return f"""Dataset loaded.
        ===============================================
        Number of training examples: {len(self.datasets["train"])}
        Number of validation examples: {len(self.datasets["val"])}
        Found {len(classes)} classes: {classes}
        ===============================================\n"""


class ImageTransformer:
    """
    Simple class object creating image transforms separately for
    training and test dataset.
    """

    @staticmethod
    def get_image_transforms(
        img_size: Tuple[int, int], phase: Union[str, None] = None
    ) -> Union[Dict[str, transforms.Compose], transforms.Compose]:
        image_transforms = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(img_size[0]),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }

        if phase is not None and phase in DATA_SPLIT:
            return image_transforms[phase]
        elif phase is None:
            return image_transforms
        else:
            raise KeyError(f"Transforms for data {phase} not found.")


class StanfordDogsImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, classes_to_labels=None):
        super().__init__(root, transform)
        self.root = root
        self.transform = transform
        self.classes_to_labels = classes_to_labels
        self.classes, self.class_to_idx = self._find_classes()

    def _find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """
        classes = sorted(
            entry.name for entry in os.scandir(self.root) if entry.is_dir()
        )

        if self.classes_to_labels is not None:
            classes = [self.classes_to_labels[class_name] for class_name in classes]

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {self.root}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class DatasetFromSubset(StanfordDogsImageDataset):
    def __init__(self, subset, root: str, transform=None, classes_to_labels=None):
        super().__init__(root, transform, classes_to_labels)
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
