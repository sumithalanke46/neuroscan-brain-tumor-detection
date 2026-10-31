"""
NeuroScan-AI: Evaluation and Visualization Module
Evaluates model on held-out test dataset, generates confusion matrix,
classification report, convergence curves, and prediction grids.
"""

import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from PIL import Image

from src.dataset import get_dataloaders, CLASS_NAMES, CLASS_DISPLAY_NAMES, get_transforms
from src.model import build_model


def set_plot_style():
    """Sets clean academic plotting aesthetics."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["figure.titlesize"] = 13


def plot_training_curves(metrics_file: str, save_path: str):
    """Plots training and validation loss and accuracy curves."""
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} not found. Skipping curve plot.")
        return

    with open(metrics_file, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], 'o-', color='#1f77b4', linewidth=2.2, label='Training Loss')
    ax1.plot(epochs, history["val_loss"], 's--', color='#d62728', linewidth=2.2, label='Validation Loss')
    ax1.set_title('Cross-Entropy Loss vs. Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Accuracy Curve
    ax2.plot(epochs, history["train_acc"], 'o-', color='#2ca02c', linewidth=2.2, label='Training Accuracy')
    ax2.plot(epochs, history["val_acc"], 's--', color='#ff7f0e', linewidth=2.2, label='Validation Accuracy')
    ax2.set_title('Classification Accuracy (%) vs. Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim([min(history["train_acc"] + history["val_acc"]) - 5, 102])
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {save_path}")


def plot_confusion_matrix_fig(cm: np.ndarray, labels: list, save_path: str):
    """Generates and saves high-resolution confusion matrix heatmap."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Calculate percentage annotation
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        linewidths=1.0,
        linecolor='white'
    )
    ax.set_title('NeuroScan-AI Confusion Matrix on Test Set', pad=15)
    ax.set_xlabel('Predicted Diagnostic Category', fontweight='bold', labelpad=10)
    ax.set_ylabel('Ground Truth Clinical Category', fontweight='bold', labelpad=10)
    plt.xticks(rotation=20, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_sample_predictions(samples: list, save_path: str):
    """Plots a 3x3 grid of MRI test samples with predictions and confidence."""
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    axes = axes.flatten()

    for idx, sample in enumerate(samples[:9]):
        img_path, true_label, pred_label, conf = sample
        img = Image.open(img_path).convert('RGB')
        ax = axes[idx]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        is_correct = (true_label == pred_label)
        color = '#2ca02c' if is_correct else '#d62728'
        status = "CORRECT" if is_correct else "MISCLASSIFIED"

        title = f"True: {CLASS_DISPLAY_NAMES[true_label]}\nPred: {CLASS_DISPLAY_NAMES[pred_label]} ({conf:.1f}%)\n[{status}]"
        ax.set_title(title, color=color, fontsize=9.5, fontweight='bold', pad=4)

    plt.suptitle("NeuroScan-AI: Sample Test MRI Diagnostic Predictions", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sample predictions to {save_path}")


def evaluate_model(
    data_dir: str = "data",
    model_path: str = "models/neuroscan_best_model.pt",
    outputs_dir: str = "outputs",
    batch_size: int = 16
):
    os.makedirs(outputs_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("NeuroScan-AI: Model Evaluation on Held-Out Test Set")
    print("=" * 65)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train the model first.")

    # 1. Load Data
    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=batch_size)
    print(f"Test Set: {len(test_loader.dataset)} images across {len(CLASS_NAMES)} classes.")

    # 2. Load Checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get("model_type", "transfer")
    model = build_model(model_type=model_type, num_classes=4, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_confs = []
    prediction_samples = []
    inference_times = []

    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            t0 = time.time()
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            batch_time = (time.time() - t0) * 1000.0 / images.size(0)  # ms per image
            inference_times.extend([batch_time] * images.size(0))

            confs, preds = torch.max(probs, 1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_confs.extend(confs.cpu().numpy().tolist())

            for i in range(images.size(0)):
                prediction_samples.append((
                    paths[i],
                    CLASS_NAMES[labels[i].item()],
                    CLASS_NAMES[preds[i].item()],
                    confs[i].item() * 100.0
                ))

    # Metrics calculation
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    accuracy = float((y_true == y_pred).mean() * 100.0)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    avg_latency = float(np.mean(inference_times))
    cm = confusion_matrix(y_true, y_pred)
    display_labels = [CLASS_DISPLAY_NAMES[c] for c in CLASS_NAMES]

    # Save Visualizations
    plot_training_curves(
        os.path.join(outputs_dir, "training_metrics.json"),
        os.path.join(outputs_dir, "accuracy_loss_curve.png")
    )
    plot_confusion_matrix_fig(
        cm,
        display_labels,
        os.path.join(outputs_dir, "confusion_matrix.png")
    )
    plot_sample_predictions(
        prediction_samples,
        os.path.join(outputs_dir, "sample_predictions.png")
    )

    # Compile Results Dict
    test_results = {
        "overall_accuracy": round(accuracy, 2),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_precision": round(float(precision_weighted), 4),
        "weighted_recall": round(float(recall_weighted), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "avg_inference_latency_ms": round(avg_latency, 2),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {}
    }

    for idx, cname in enumerate(CLASS_NAMES):
        test_results["per_class_metrics"][cname] = {
            "display_name": CLASS_DISPLAY_NAMES[cname],
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1_score": round(float(per_class_f1[idx]), 4),
            "support": int(per_class_supp[idx])
        }

    results_file = os.path.join(outputs_dir, "test_metrics.json")
    with open(results_file, "w") as f:
        json.dump(test_results, f, indent=4)

    # Print Formatted Evaluation Report
    print("\n" + "=" * 65)
    print("NEUROSCAN-AI FINAL PERFORMANCE EVALUATION REPORT")
    print("=" * 65)
    print(f"Overall Test Accuracy: {accuracy:.2f}%")
    print(f"Macro F1-Score:        {f1_macro:.4f}  | Weighted F1-Score: {f1_weighted:.4f}")
    print(f"Average Latency:       {avg_latency:.2f} ms / scan")
    print("-" * 65)
    print(f"{'Class':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)
    for cname in CLASS_NAMES:
        m = test_results["per_class_metrics"][cname]
        print(f"{m['display_name']:<22} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f} | {m['support']:<8}")
    print("-" * 65)
    print(f"Saved evaluation metrics to: {results_file}")
    return test_results


if __name__ == "__main__":
    evaluate_model()
