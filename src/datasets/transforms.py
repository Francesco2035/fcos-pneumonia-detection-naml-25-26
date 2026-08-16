from torchvision import transforms


def get_train_transforms(image_size=224):

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),

        transforms.ColorJitter(
            brightness=0.2
        ),

        transforms.ToTensor(),
    ])


def get_test_transforms(image_size=224):

    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size)
        ),

        transforms.ToTensor(),
    ])