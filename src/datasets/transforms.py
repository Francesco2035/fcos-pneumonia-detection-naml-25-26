import torch
from torchvision.transforms import v2


from torchvision.transforms import v2


def get_train_transforms(image_size=224):

    return v2.Compose([
        v2.Resize((image_size, image_size)),

        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),

        v2.ColorJitter(
            brightness=0.2
        ),

        v2.Grayscale(
            num_output_channels=3
        ),

        v2.ToDtype(
            torch.float32,
            scale=True
        ),
    ])

def get_test_transforms(image_size=224):

    return v2.Compose([
        v2.Resize((image_size, image_size)),

        v2.Grayscale(
            num_output_channels=3
        ),

        v2.ToDtype(
            torch.float32,
            scale=True
        ),
    ])