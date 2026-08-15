import os
import torch

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

from sklearn.metrics import confusion_matrix, classification_report

from src.datasets.chest_xray import ChestXRayDataModule
from src.train import train

from src.models.resnet import ResNet50
from src.models.pretrained import get_pretrained_resnet50


# =========================================================
# Experiment configuration
# =========================================================

MODEL_NAME = "scratch"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3

DATA_DIR = "data/chest_xray"
CHECKPOINT_DIR = "checkpoints"
RUNS_DIR = "runs"


# =========================================================
# Model factory
# =========================================================

def build_model(model_name):

    if model_name == "scratch":

        model = ResNet50(
            img_channels=3,
            num_classes=2,
        )

    elif model_name == "pretrained":

        model = get_pretrained_resnet50(
            num_classes=2,
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model


# =========================================================
# Evaluation
# =========================================================

def evaluate_predictions(
    model,
    loader,
    device,
):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    return all_labels, all_predictions


# =========================================================
# Main
# =========================================================

def main():

    # =====================================================
    # Device
    # =====================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # =====================================================
    # Dataset
    # =====================================================

    data = ChestXRayDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = data.get_dataloaders()

    print("\n=========================")
    print("DATASET")
    print("=========================")

    print(
        f"Train samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    print(
        "\nTrain classes:",
        Counter(
            train_loader.dataset.targets
        ),
    )

    print(
        "Validation classes:",
        Counter(
            val_loader.dataset.targets
        ),
    )

    print(
        "Test classes:",
        Counter(
            test_loader.dataset.targets
        ),
    )

    print(
        "\nClass mapping:",
        train_loader.dataset.class_to_idx,
    )

    # =====================================================
    # Model
    # =====================================================

    model = build_model(MODEL_NAME)

    model = model.to(device)

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\n=========================")
    print("MODEL")
    print("=========================")

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Parameters: "
        f"{num_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    # =====================================================
    # Loss
    # =====================================================

    criterion = nn.CrossEntropyLoss()

    # =====================================================
    # Optimizer
    # =====================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # =====================================================
    # Experiment
    # =====================================================

    model_label = (
        f"resnet50_{MODEL_NAME}_chest_xray"
    )

    checkpoint_dir = CHECKPOINT_DIR

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    writer = SummaryWriter(
        log_dir=f"{RUNS_DIR}/{model_label}"
    )

    print("\n=========================")
    print("EXPERIMENT")
    print("=========================")

    print(
        f"Experiment: {model_label}"
    )

    print(
        f"Learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs: "
        f"{EPOCHS}"
    )

    # =====================================================
    # Training
    # =====================================================

    print("\n=========================")
    print("TRAINING")
    print("=========================")

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        writer=writer,
        model_label=model_label,
        checkpoint_dir=checkpoint_dir,
    )

    # =====================================================
    # Load BEST model
    # =====================================================

    best_model_path = (
        f"{checkpoint_dir}/"
        f"{model_label}_best.pth"
    )

    print("\n=========================")
    print("BEST MODEL")
    print("=========================")

    print(
        f"Loading: "
        f"{best_model_path}"
    )

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    best_epoch = (
        checkpoint["epoch"] + 1
    )

    best_val_loss = (
        checkpoint["val_loss"]
    )

    print(
        f"Best epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_val_loss:.4f}"
    )

    # =====================================================
    # Final TEST evaluation
    # =====================================================

    labels, predictions = (
        evaluate_predictions(
            model=model,
            loader=test_loader,
            device=device,
        )
    )

    print("\n=========================")
    print("TEST RESULTS")
    print("=========================")

    cm = confusion_matrix(
        labels,
        predictions,
    )

    print("\nConfusion Matrix:")

    print(cm)

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            labels,
            predictions,
            target_names=[
                "NORMAL",
                "PNEUMONIA",
            ],
            digits=4,
        )
    )

    # =====================================================
    # Cleanup
    # =====================================================

    writer.close()


# =========================================================
# Entry point
# =========================================================

if __name__ == "__main__":
    main()