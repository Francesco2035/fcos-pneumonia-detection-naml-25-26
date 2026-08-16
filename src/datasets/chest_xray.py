import os

from torch.utils.data import DataLoader
from torchvision import datasets

from src.datasets.transforms import (
    get_train_transforms,
    get_test_transforms,
)


class ChestXRayDataModule:

    def __init__(
        self,
        data_dir,
        batch_size=32,
        image_size=224,
    ):

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size

        # ---------------------------------
        # Transforms
        # ---------------------------------

        self.train_transform = get_train_transforms(
            image_size=image_size
        )

        self.test_transform = get_test_transforms(
            image_size=image_size
        )

    # ---------------------------------
    # Datasets
    # ---------------------------------

    def get_datasets(self):

        train_dataset = datasets.ImageFolder(
            root=os.path.join(
                self.data_dir,
                "train",
            ),
            transform=self.train_transform,
        )

        val_dataset = datasets.ImageFolder(
            root=os.path.join(
                self.data_dir,
                "val",
            ),
            transform=self.test_transform,
        )

        test_dataset = datasets.ImageFolder(
            root=os.path.join(
                self.data_dir,
                "test",
            ),
            transform=self.test_transform,
        )

        return (
            train_dataset,
            val_dataset,
            test_dataset,
        )

    # ---------------------------------
    # DataLoaders
    # ---------------------------------

    def get_dataloaders(self):

        train_dataset, val_dataset, test_dataset = (
            self.get_datasets()
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return (
            train_loader,
            val_loader,
            test_loader,
        )