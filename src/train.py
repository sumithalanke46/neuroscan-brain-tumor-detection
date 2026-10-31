"""
NeuroScan-AI: Training Module
Trains the deep transfer learning neural network with Cross-Entropy Loss,
AdamW optimizer, Cosine Annealing learning rate schedule, and best-model checkpointing.
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset import get_dataloaders
from src.model import build_model, count_parameters


def train_model(
    data_dir: str = "data",
    models_dir: str = "models",
    outputs_dir: str = "outputs",
    model_type: str = "transfer",
    epochs: int = 8,
    batch_size: int = 16,
    lr: float = 1e-3,
    weight_decay: float = 1e-4
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"NeuroScan-AI: Training Pipeline Initiated")
    print(f"Device: {device} | Model Type: {model_type} | Epochs: {epochs} | Batch Size: {batch_size}")
    print("=" * 65)

    # 1. DataLoaders
    train_loader, val_loader, _ = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    print(f"Train Batches: {len(train_loader)} ({len(train_loader.dataset)} images)")
    print(f"Validation Batches: {len(val_loader)} ({len(val_loader.dataset)} images)")

    # 2. Build Model
    model = build_model(model_type=model_type, num_classes=4, pretrained=True)
    model = model.to(device)
    params = count_parameters(model)
    print(f"Model Parameters: {params['total']:,} total, {params['trainable']:,} trainable.")

    # 3. Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr_history": []
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Training Phase
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 2))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 2))
        history["lr_history"].append(current_lr)

        epoch_duration = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - {epoch_duration:.1f}s | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}% | LR: {current_lr:.2e}")

        # Checkpoint Best Model
        best_path = os.path.join(models_dir, "neuroscan_best_model.pt")
        if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < best_val_loss):
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "model_type": model_type
            }, best_path)
            print(f"  -> Saved new best model to {best_path} (Val Acc: {best_val_acc:.2f}%)")

    total_time = time.time() - start_time
    history["total_training_time_seconds"] = round(total_time, 2)
    history["best_val_acc"] = best_val_acc
    history["best_val_loss"] = best_val_loss
    history["epochs"] = epochs
    history["model_type"] = model_type

    metrics_file = os.path.join(outputs_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(history, f, indent=4)
    print("=" * 65)
    print(f"Training Complete! Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Peak Validation Accuracy: {best_val_acc:.2f}% | Metrics logged to {metrics_file}")
    print("=" * 65)
    return history


if __name__ == "__main__":
    train_model(epochs=8, batch_size=16)
