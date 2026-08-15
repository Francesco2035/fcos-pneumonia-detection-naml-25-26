import os
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


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
        # Transform usato SOLO per calcolare
        # mean e std del training set
        # ---------------------------------

        self.stats_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # ---------------------------------
        # Calcola statistiche del train set
        # ---------------------------------

        self.mean, self.std = self._compute_mean_std()

        print(f"Dataset mean: {self.mean}")
        print(f"Dataset std: {self.std}")

        # ---------------------------------
        # Transform di training
        # ---------------------------------

        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean,
                std=self.std,
            ),
        ])

        # ---------------------------------
        # Transform validation/test
        # ---------------------------------

        self.test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean,
                std=self.std,
            ),
        ])

    def _compute_mean_std(self):

        dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"),
            transform=self.stats_transform,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        mean = torch.zeros(3)
        std = torch.zeros(3)

        total_images = 0

        for images, _ in loader:

            batch_size = images.size(0)

            # [B, C, H, W]
            # → [B, C, H*W]

            images = images.view(
                batch_size,
                images.size(1),
                -1,
            )

            # Media per immagine e canale
            batch_mean = images.mean(dim=2)
            batch_std = images.std(dim=2)

            mean += batch_mean.sum(dim=0)
            std += batch_std.sum(dim=0)

            total_images += batch_size

        mean /= total_images
        std /= total_images

        return mean.tolist(), std.tolist()

    def get_datasets(self):

        train_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "train"),
            transform=self.train_transform,
        )

        val_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "val"),
            transform=self.test_transform,
        )

        test_dataset = datasets.ImageFolder(
            root=os.path.join(self.data_dir, "test"),
            transform=self.test_transform,
        )

        return train_dataset, val_dataset, test_dataset

    def get_dataloaders(self):

        train, val, test = self.get_datasets()

        train_loader = DataLoader(
            train,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val,
            batch_size=self.batch_size,
            shuffle=False,
        )

        test_loader = DataLoader(
            test,
            batch_size=self.batch_size,
            shuffle=False,
        )

        return train_loader, val_loader, test_loader