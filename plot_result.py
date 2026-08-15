import os
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_DIR = "runs/resnet50_scratch_chest_xray"
OUTPUT_DIR = "results/figures"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_scalar(event_accumulator, tag):

    events = event_accumulator.Scalars(tag)

    epochs = [
        event.step + 1
        for event in events
    ]

    values = [
        event.value
        for event in events
    ]

    return epochs, values


# =========================
# Load latest TensorBoard run
# =========================

event_files = list(
    Path(RUN_DIR).glob("events.out.tfevents.*")
)

if not event_files:

    raise RuntimeError(
        f"Nessun event file trovato in {RUN_DIR}"
    )


# Prende il file TensorBoard modificato più recentemente
latest_event_file = max(
    event_files,
    key=lambda path: path.stat().st_mtime,
)

print(
    "Uso event file:",
    latest_event_file,
)


ea = EventAccumulator(
    str(latest_event_file)
)

ea.Reload()


# =========================
# TensorBoard tags
# =========================

tags = ea.Tags()["scalars"]

train_accuracy_tag = (
    "resnet50_scratch_chest_xray/Train/Accuracy"
)

val_accuracy_tag = (
    "resnet50_scratch_chest_xray/Validation/Accuracy"
)

train_loss_tag = (
    "resnet50_scratch_chest_xray/Train/Loss"
)

val_loss_tag = (
    "resnet50_scratch_chest_xray/Validation/Loss"
)


# Controlla che i tag esistano
required_tags = [
    train_accuracy_tag,
    val_accuracy_tag,
    train_loss_tag,
    val_loss_tag,
]

missing_tags = [
    tag
    for tag in required_tags
    if tag not in tags
]

if missing_tags:

    raise RuntimeError(
        "Mancano i seguenti TensorBoard tags:\n"
        + "\n".join(missing_tags)
    )


# =========================
# Accuracy
# =========================

train_epochs, train_accuracy = load_scalar(
    ea,
    train_accuracy_tag,
)

val_epochs, val_accuracy = load_scalar(
    ea,
    val_accuracy_tag,
)


plt.figure(figsize=(8, 5))

plt.plot(
    train_epochs,
    train_accuracy,
    marker="o",
    label="Training",
)

plt.plot(
    val_epochs,
    val_accuracy,
    marker="o",
    label="Validation",
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "Training and Validation Accuracy"
)

plt.xticks(
    range(
        1,
        max(train_epochs + val_epochs) + 1
    )
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    Path(OUTPUT_DIR) / "accuracy.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================
# Loss
# =========================

train_epochs, train_loss = load_scalar(
    ea,
    train_loss_tag,
)

val_epochs, val_loss = load_scalar(
    ea,
    val_loss_tag,
)


plt.figure(figsize=(8, 5))

plt.plot(
    train_epochs,
    train_loss,
    marker="o",
    label="Training",
)

plt.plot(
    val_epochs,
    val_loss,
    marker="o",
    label="Validation",
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "Training and Validation Loss"
)

plt.xticks(
    range(
        1,
        max(train_epochs + val_epochs) + 1
    )
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    Path(OUTPUT_DIR) / "loss.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# =========================
# Done
# =========================

print("\nGrafici creati!")

print(
    "Accuracy:",
    Path(OUTPUT_DIR) / "accuracy.png"
)

print(
    "Loss:",
    Path(OUTPUT_DIR) / "loss.png"
)