import random


def get_positive_negative_indices(dataset):
    """
    Separate dataset indices into positive and negative images.

    Positive image:
        at least one ground-truth bounding box.

    Negative image:
        no ground-truth bounding boxes.
    """

    positive_indices = []
    negative_indices = []

    for index in range(len(dataset)):

        # Get the image id without applying transforms.
        patient_id = dataset.image_paths[index].stem

        boxes = dataset.annotations[patient_id]["boxes"]

        if len(boxes) > 0:
            positive_indices.append(index)
        else:
            negative_indices.append(index)

    return positive_indices, negative_indices


def create_train_val_split(
    dataset,
    val_ratio=0.2,
    seed=42,
):
    """
    Create a reproducible train/validation split.

    The split is stratified by the presence of GT boxes:
        - positive images
        - negative images

    Returns:
        train_indices
        val_indices
    """

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(
            "val_ratio must be between 0 and 1."
        )

    positive_indices, negative_indices = (
        get_positive_negative_indices(dataset)
    )

    rng = random.Random(seed)

    # Shuffle each group independently.
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)

    # Number of validation samples per group.
    num_positive_val = round(
        len(positive_indices) * val_ratio
    )

    num_negative_val = round(
        len(negative_indices) * val_ratio
    )

    # Split positive samples.
    val_positive = positive_indices[
        :num_positive_val
    ]

    train_positive = positive_indices[
        num_positive_val:
    ]

    # Split negative samples.
    val_negative = negative_indices[
        :num_negative_val
    ]

    train_negative = negative_indices[
        num_negative_val:
    ]

    # Combine the two groups.
    train_indices = (
        train_positive
        + train_negative
    )

    val_indices = (
        val_positive
        + val_negative
    )

    # Shuffle final sets.
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def print_split_statistics(
    dataset,
    train_indices,
    val_indices,
):
    """
    Print useful statistics for the train/validation split.
    """

    positive_indices, negative_indices = (
        get_positive_negative_indices(dataset)
    )

    positive_set = set(positive_indices)
    negative_set = set(negative_indices)

    train_positive = sum(
        index in positive_set
        for index in train_indices
    )

    train_negative = sum(
        index in negative_set
        for index in train_indices
    )

    val_positive = sum(
        index in positive_set
        for index in val_indices
    )

    val_negative = sum(
        index in negative_set
        for index in val_indices
    )

    print("\nTrain/Validation split")
    print("----------------------")

    print(
        f"Total images:       {len(dataset)}"
    )

    print(
        f"Train images:       {len(train_indices)}"
    )

    print(
        f"Validation images:  {len(val_indices)}"
    )

    print()

    print(
        f"Train positive:     {train_positive}"
    )

    print(
        f"Train negative:     {train_negative}"
    )

    print()

    print(
        f"Validation positive: {val_positive}"
    )

    print(
        f"Validation negative: {val_negative}"
    )

    print()

    if len(train_indices) > 0:
        train_positive_ratio = (
            train_positive / len(train_indices)
        )
    else:
        train_positive_ratio = 0.0

    if len(val_indices) > 0:
        val_positive_ratio = (
            val_positive / len(val_indices)
        )
    else:
        val_positive_ratio = 0.0

    print(
        f"Train positive ratio: "
        f"{train_positive_ratio:.4f}"
    )

    print(
        f"Validation positive ratio: "
        f"{val_positive_ratio:.4f}"
    )