from src.datasets.RSNAPneumoniaDataset import (
    RSNAPneumoniaDataset,
)

from src.datasets.transforms import (
    get_train_transforms,
)

import random
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# =========================================================
# PATHS
# =========================================================

CSV_PATH = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_labels.csv"
)

TRAIN_DCM_PATH = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_images"
)


# =========================================================
# CREATE DATASET
# =========================================================

def create_dataset(transform=None):

    return RSNAPneumoniaDataset(
        dcm_path=TRAIN_DCM_PATH,
        csv_path=CSV_PATH,
        transform=transform,
    )


# =========================================================
# FIND POSITIVE IMAGES
# =========================================================

def get_positive_indices(dataset):

    positive_indices = []

    for i in range(len(dataset)):

        patient_id = dataset.image_paths[i].stem

        if len(
            dataset.annotations[patient_id]["boxes"]
        ) > 0:

            positive_indices.append(i)

    return positive_indices


# =========================================================
# CHECK SINGLE DATASET ITEM
# =========================================================

def check_dataset_item(
    dataset,
    index=0,
):

    image, target = dataset[index]

    print("\n================ DATASET ITEM ================")

    print("Index:", index)

    print("Image type:")
    print(type(image))

    print("Image shape:")
    print(image.shape)

    print("Image dtype:")
    print(image.dtype)

    print("Image min:")
    print(image.min().item())

    print("Image max:")
    print(image.max().item())

    print("\nBoxes:")
    print(target["boxes"])

    print("\nLabels:")
    print(target["labels"])

    # -----------------------------------------------------
    # Check boxes / labels
    # -----------------------------------------------------

    assert len(
        target["boxes"]
    ) == len(
        target["labels"]
    )

    # -----------------------------------------------------
    # Check boxes
    # -----------------------------------------------------

    boxes = target["boxes"]

    h, w = boxes.canvas_size

    assert torch.all(
        boxes[:, 0] >= 0
    )

    assert torch.all(
        boxes[:, 1] >= 0
    )

    assert torch.all(
        boxes[:, 2] <= w
    )

    assert torch.all(
        boxes[:, 3] <= h
    )

    assert torch.all(
        boxes[:, 0] < boxes[:, 2]
    )

    assert torch.all(
        boxes[:, 1] < boxes[:, 3]
    )

    print("\n✓ Dataset item test passed")


# =========================================================
# CHECK DATALOADER
# =========================================================

def check_dataloader(
    batch_size=4,
):

    dataset = create_dataset(
        transform=get_train_transforms(
            image_size=224
        )
    )

    loader = dataset.get_dataloader(
        batch_size=batch_size,
        shuffle=True,
    )

    images, targets = next(
        iter(loader)
    )

    print("\n================ DATALOADER ================")

    print("Batch size:", batch_size)

    print("\nImages")

    print("type:")
    print(type(images))

    print("shape:")
    print(images.shape)

    print("dtype:")
    print(images.dtype)

    print("\nTargets")

    print("type:")
    print(type(targets))

    print("number of targets:")
    print(len(targets))

    # -----------------------------------------------------
    # Check image batch
    # -----------------------------------------------------

    assert images.shape[0] == batch_size

    assert images.shape[1:] == (
        1,
        224,
        224,
    )

    # -----------------------------------------------------
    # Check targets
    # -----------------------------------------------------

    assert len(targets) == batch_size

    for i, target in enumerate(targets):

        print(f"\nTarget {i}")

        print(
            "boxes shape:",
            target["boxes"].shape,
        )

        print(
            "labels shape:",
            target["labels"].shape,
        )

        assert len(
            target["boxes"]
        ) == len(
            target["labels"]
        )

    print("\n✓ DataLoader test passed")


# =========================================================
# DRAW BOXES
# =========================================================

def draw_boxes(
    ax,
    image,
    target,
):

    ax.imshow(
        image,
        cmap="gray",
    )

    for box, label in zip(
        target["boxes"],
        target["labels"],
    ):

        x1, y1, x2, y2 = box.tolist()

        rectangle = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
        )

        ax.add_patch(rectangle)

        ax.text(
            x1,
            y1,
            str(label.item()),
            fontsize=10,
        )

    ax.axis("off")


# =========================================================
# CHECK TRANSFORMED TARGET
# =========================================================

def check_transformed_target(
    original_target,
    transformed_target,
):

    # -----------------------------------------------------
    # Number boxes == number labels
    # -----------------------------------------------------

    assert len(
        original_target["boxes"]
    ) == len(
        original_target["labels"]
    )

    assert len(
        transformed_target["boxes"]
    ) == len(
        transformed_target["labels"]
    )

    # -----------------------------------------------------
    # Labels unchanged
    # -----------------------------------------------------

    assert torch.equal(
        original_target["labels"],
        transformed_target["labels"],
    )

    # -----------------------------------------------------
    # Valid boxes
    # -----------------------------------------------------

    for target in [
        original_target,
        transformed_target,
    ]:

        boxes = target["boxes"]

        h, w = boxes.canvas_size

        assert torch.all(
            boxes[:, 0] >= 0
        )

        assert torch.all(
            boxes[:, 1] >= 0
        )

        assert torch.all(
            boxes[:, 2] <= w
        )

        assert torch.all(
            boxes[:, 3] <= h
        )

        assert torch.all(
            boxes[:, 0] < boxes[:, 2]
        )

        assert torch.all(
            boxes[:, 1] < boxes[:, 3]
        )


# =========================================================
# VISUALIZE AUGMENTATIONS
# =========================================================

def visualize_augmentations(
    num_images=10,
    seed=43,
):

    # -----------------------------------------------------
    # Dataset originale
    # -----------------------------------------------------

    dataset = create_dataset(
        transform=None
    )

    # -----------------------------------------------------
    # Dataset trasformato
    # -----------------------------------------------------

    transformed_dataset = create_dataset(
        transform=get_train_transforms(
            image_size=224
        )
    )

    # -----------------------------------------------------
    # Positive images
    # -----------------------------------------------------

    positive_indices = get_positive_indices(
        dataset
    )

    print(
        "\nNumero immagini positive:",
        len(positive_indices),
    )

    # -----------------------------------------------------
    # Random selection
    # -----------------------------------------------------

    random.seed(seed)

    num_images = min(
        num_images,
        len(positive_indices),
    )

    selected_indices = random.sample(
        positive_indices,
        num_images,
    )

    print(
        "Indici selezionati:",
        selected_indices,
    )

    # -----------------------------------------------------
    # Figure
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
    )

    current = 0

    # -----------------------------------------------------
    # Show image
    # -----------------------------------------------------

    def show_image(position):

        index = selected_indices[position]

        # ---------------------------------------------
        # Original
        # ---------------------------------------------

        original_image, original_target = (
            dataset[index]
        )

        # ---------------------------------------------
        # Transformed
        # ---------------------------------------------

        transformed_image, transformed_target = (
            transformed_dataset[index]
        )

        # ---------------------------------------------
        # Checks
        # ---------------------------------------------

        check_transformed_target(
            original_target,
            transformed_target,
        )

        # ---------------------------------------------
        # Remove channel dimension
        # ---------------------------------------------

        original_image = (
            original_image.squeeze(0)
        )

        transformed_image = (
            transformed_image.squeeze(0)
        )

        # ---------------------------------------------
        # Clear axes
        # ---------------------------------------------

        axes[0].clear()
        axes[1].clear()

        # ---------------------------------------------
        # Draw
        # ---------------------------------------------

        draw_boxes(
            axes[0],
            original_image,
            original_target,
        )

        draw_boxes(
            axes[1],
            transformed_image,
            transformed_target,
        )

        # ---------------------------------------------
        # Titles
        # ---------------------------------------------

        patient_id = (
            dataset.image_paths[index].stem
        )

        axes[0].set_title(
            f"Original | "
            f"{position + 1}/{num_images}",
            fontsize=11,
        )

        axes[1].set_title(
            "Transformed",
            fontsize=11,
        )

        fig.suptitle(
            patient_id,
            fontsize=9,
        )

        fig.tight_layout()

        fig.canvas.draw_idle()

    # -----------------------------------------------------
    # Keyboard
    # -----------------------------------------------------

    def on_key(event):

        nonlocal current

        if event.key in [
            "right",
            "n",
        ]:

            current = (
                current + 1
            ) % num_images

            show_image(current)

        elif event.key in [
            "left",
            "p",
        ]:

            current = (
                current - 1
            ) % num_images

            show_image(current)

        elif event.key == "q":

            plt.close(fig)

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    show_image(current)

    plt.show()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Test Dataset
    # -----------------------------------------------------
    """

    dataset = create_dataset(
        transform=get_train_transforms(
            image_size=224
        )
    )
    
    check_dataset_item(
        dataset,
        index=0,
    )
    """
    # -----------------------------------------------------
    # Test DataLoader
    # -----------------------------------------------------

    check_dataloader(
        batch_size=4,
    )

    # -----------------------------------------------------
    # Visual test
    # -----------------------------------------------------

    """visualize_augmentations(
        num_images=10,
        seed=43,
    )"""