from src.datasets.RSNAPneumoniaDataset import RSNAPneumoniaDataset
from src.datasets.transforms import get_train_transforms

import random
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# =========================================================
# Paths
# =========================================================

csv_path = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_labels.csv"
)

train_dcm_path = (
    "data/rsna-pneumonia-detection-challenge/"
    "stage_2_train_images"
)


# =========================================================
# Dataset originale
# =========================================================

dataset = RSNAPneumoniaDataset(
    dcm_path=train_dcm_path,
    csv_path=csv_path,
    transform=None,
)


# =========================================================
# Trova tutte le immagini positive
# =========================================================

positive_indices = []

for i in range(len(dataset)):

    patient_id = dataset.image_paths[i].stem

    if len(dataset.annotations[patient_id]["boxes"]) > 0:
        positive_indices.append(i)


print("Numero immagini positive:", len(positive_indices))


# =========================================================
# Seleziona 10 immagini casuali
# =========================================================

random.seed(43)

num_images = min(
    10,
    len(positive_indices)
)

selected_indices = random.sample(
    positive_indices,
    num_images
)

print("Indici selezionati:")
print(selected_indices)


# =========================================================
# Train transforms
# =========================================================

transform = get_train_transforms(
    image_size=224
)


# =========================================================
# Dataset trasformato
# =========================================================

transformed_dataset = RSNAPneumoniaDataset(
    dcm_path=train_dcm_path,
    csv_path=csv_path,
    transform=transform,
)


# =========================================================
# Funzione per disegnare bounding boxes
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
            f"{label.item()}",
            fontsize=10,
            verticalalignment="bottom",
        )

    ax.axis("off")


# =========================================================
# Funzione di controllo
# =========================================================

def check_target(
    original_target,
    transformed_target,
):

    # -----------------------------------------------------
    # Numero box = numero label
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
    # Le labels non devono cambiare
    # -----------------------------------------------------

    assert torch.equal(
        original_target["labels"],
        transformed_target["labels"],
    )

    # -----------------------------------------------------
    # Controllo validità bounding boxes
    # -----------------------------------------------------

    for target in [
        original_target,
        transformed_target,
    ]:

        boxes = target["boxes"]

        h, w = boxes.canvas_size

        # coordinate >= 0
        assert torch.all(
            boxes[:, 0] >= 0
        )

        assert torch.all(
            boxes[:, 1] >= 0
        )

        # coordinate dentro il canvas
        assert torch.all(
            boxes[:, 2] <= w
        )

        assert torch.all(
            boxes[:, 3] <= h
        )

        # box non degeneri
        assert torch.all(
            boxes[:, 0] < boxes[:, 2]
        )

        assert torch.all(
            boxes[:, 1] < boxes[:, 3]
        )


# =========================================================
# Figura
# =========================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 6),
)


current = 0


# =========================================================
# Mostra immagine
# =========================================================

def show_image(position):

    index = selected_indices[position]

    # -----------------------------------------------------
    # Originale
    # -----------------------------------------------------

    original_image, original_target = (
        dataset[index]
    )

    # -----------------------------------------------------
    # Trasformata
    # -----------------------------------------------------

    transformed_image, transformed_target = (
        transformed_dataset[index]
    )

    # -----------------------------------------------------
    # Check
    # -----------------------------------------------------

    check_target(
        original_target,
        transformed_target,
    )

    # -----------------------------------------------------
    # Rimuove il channel singleton
    # [1, H, W] -> [H, W]
    # -----------------------------------------------------

    original_image = original_image.squeeze(0)

    transformed_image = transformed_image.squeeze(0)

    # -----------------------------------------------------
    # Pulisce i pannelli
    # -----------------------------------------------------

    axes[0].clear()
    axes[1].clear()

    # -----------------------------------------------------
    # Disegna originale
    # -----------------------------------------------------

    draw_boxes(
        axes[0],
        original_image,
        original_target,
    )

    # -----------------------------------------------------
    # Disegna trasformata
    # -----------------------------------------------------

    draw_boxes(
        axes[1],
        transformed_image,
        transformed_target,
    )

    # -----------------------------------------------------
    # Patient ID
    # -----------------------------------------------------

    patient_id = (
        dataset.image_paths[index].stem
    )

    # -----------------------------------------------------
    # Titoli
    # -----------------------------------------------------

    axes[0].set_title(
        f"Original  |  {position + 1}/{num_images}",
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


# =========================================================
# Gestione tastiera
# =========================================================

def on_key(event):

    global current

    # -----------------------------------------------------
    # Avanti
    # -----------------------------------------------------

    if event.key in [
        "right",
        "n",
    ]:

        current = (
            current + 1
        ) % num_images

        show_image(current)

    # -----------------------------------------------------
    # Indietro
    # -----------------------------------------------------

    elif event.key in [
        "left",
        "p",
    ]:

        current = (
            current - 1
        ) % num_images

        show_image(current)

    # -----------------------------------------------------
    # Esci
    # -----------------------------------------------------

    elif event.key == "q":

        plt.close(fig)


fig.canvas.mpl_connect(
    "key_press_event",
    on_key,
)


# =========================================================
# Mostra la prima immagine
# =========================================================

show_image(current)

plt.show()