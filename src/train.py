import torch
import os


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    writer=None,
    epoch=0,
    model_label=None,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total

    if writer is not None:
        writer.add_scalar(
            f"{model_label}/Train/Loss",
            epoch_loss,
            epoch,
        )

        writer.add_scalar(
            f"{model_label}/Train/Accuracy",
            epoch_accuracy,
            epoch,
        )

    return epoch_loss, epoch_accuracy


def validate(
    model,
    val_loader,
    criterion,
    device,
):
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = correct / total

    return val_loss, val_accuracy


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs,
    writer=None,
    model_label=None,
    checkpoint_dir="checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(epochs):

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            writer=writer,
            epoch=epoch,
            model_label=model_label,
        )

        val_loss, val_accuracy = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        if writer is not None:
            writer.add_scalar(
                f"{model_label}/Validation/Loss",
                val_loss,
                epoch,
            )

            writer.add_scalar(
                f"{model_label}/Validation/Accuracy",
                val_accuracy,
                epoch,
            )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # Salva sempre l'ultimo stato
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            path=f"{checkpoint_dir}/{model_label}_last.pth",
        )

        # Salva il migliore
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                path=f"{checkpoint_dir}/{model_label}_best.pth",
            )

            print("  → New best model!")


def save_checkpoint(
    model,
    optimizer,
    epoch,
    val_loss,
    path,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
    }

    torch.save(checkpoint, path)