"""
NeuroScan-AI: Academic Diagrams Generator
Generates high-resolution academic figures for the project report:
- Figure 1: Overall System Flowchart
- Figure 2: Deep Transfer Learning Architecture Diagram
- Figure 3: Implementation Pipeline Flowchart
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)


def create_system_flowchart(save_path: str):
    """Generates Figure 1: Overall system flowchart matching academic standard."""
    fig, ax = plt.subplots(figsize=(8.5, 9.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Styles
    box_blue = dict(boxstyle="square,pad=0.5", facecolor="#d0e1fd", edgecolor="#185abc", linewidth=1.5)
    box_orange = dict(boxstyle="square,pad=0.5", facecolor="#feeed9", edgecolor="#b06000", linewidth=1.5)
    box_green = dict(boxstyle="square,pad=0.5", facecolor="#ceead6", edgecolor="#137333", linewidth=1.5)
    box_purple = dict(boxstyle="square,pad=0.5", facecolor="#f3e8fd", edgecolor="#7627bb", linewidth=1.5)
    arrow_props = dict(arrowstyle="->", color="#3c4043", lw=1.8, mutation_scale=15)

    # 1. Dataset Box
    ax.text(5, 11.2, "Brain Tumor MRI Dataset\n(Glioma, Meningioma, Pituitary, Normal)", 
            ha="center", va="center", bbox=box_blue, fontsize=10, fontweight="bold")
    ax.annotate("", xy=(5, 10.3), xytext=(5, 10.7), arrowprops=arrow_props)

    # 2. Preprocessing Box
    ax.text(5, 9.7, "Data Ingestion & Preprocessing\n- Grayscale/RGB Standardization\n- Dynamic Resizing to 224 x 224\n- ImageNet Normalization (μ, σ)", 
            ha="center", va="center", bbox=box_orange, fontsize=9.5)
    ax.annotate("", xy=(5, 8.7), xytext=(5, 9.1), arrowprops=arrow_props)

    # 3. Data Augmentation
    ax.text(5, 8.1, "Data Augmentation Pipeline\n- Random Horizontal Flip (p=0.5)\n- Random Rotation (±15°)\n- Color & Contrast Jittering", 
            ha="center", va="center", bbox=box_orange, fontsize=9.5)
    ax.annotate("", xy=(5, 7.1), xytext=(5, 7.5), arrowprops=arrow_props)

    # 4. Hybrid Modeling Block (Outer dashed container)
    rect = patches.FancyBboxPatch((1.2, 3.4), 7.6, 3.3, boxstyle="round,pad=0.2",
                                  facecolor="#f8f9fa", edgecolor="#5f6368", linestyle="--", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(5, 6.4, "Deep Transfer Learning Engine (MobileNetV2)", 
            ha="center", va="center", fontsize=11, fontweight="bold", color="#1a73e8")

    # Inner boxes
    ax.text(5, 5.5, "Inverted Residual Backbone\n- 53 Convolutional / Depthwise Separable Layers\n- Linear Bottleneck Feature Representation (1280-D)", 
            ha="center", va="center", bbox=box_blue, fontsize=9)
    ax.annotate("", xy=(5, 4.7), xytext=(5, 5.0), arrowprops=arrow_props)

    ax.text(5, 4.1, "Custom Multiclass Diagnostic Classifier Head\n- Dropout(0.3) -> Linear(1280, 256) -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear(256, 4)", 
            ha="center", va="center", bbox=box_purple, fontsize=8.5)

    ax.annotate("", xy=(5, 2.7), xytext=(5, 3.4), arrowprops=arrow_props)

    # 5. Softmax & Output
    ax.text(5, 2.1, "Softmax Multi-Class Probability Distribution\nP(Glioma), P(Meningioma), P(Pituitary), P(Normal)", 
            ha="center", va="center", bbox=box_green, fontsize=9.5, fontweight="bold")
    ax.annotate("", xy=(5, 1.1), xytext=(5, 1.5), arrowprops=arrow_props)

    # 6. Evaluation & UI Delivery
    ax.text(5, 0.6, "Clinical Diagnostic Verification & Web Demonstration\n- Performance Metrics (Accuracy, Precision, Recall, F1)\n- Streamlit Interactive Web Application (app.py)", 
            ha="center", va="center", bbox=box_blue, fontsize=9.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {save_path}")


def create_architecture_diagram(save_path: str):
    """Generates Figure 2: Detailed Deep Learning Architecture Diagram."""
    fig, ax = plt.subplots(figsize=(8.5, 9.0), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis('off')

    box_conv = dict(boxstyle="square,pad=0.4", facecolor="#e8f0fe", edgecolor="#1a73e8", linewidth=1.5)
    box_res = dict(boxstyle="square,pad=0.4", facecolor="#fef7e0", edgecolor="#f29900", linewidth=1.5)
    box_dense = dict(boxstyle="square,pad=0.4", facecolor="#fce8e6", edgecolor="#d93025", linewidth=1.5)
    box_norm = dict(boxstyle="square,pad=0.4", facecolor="#e6f4ea", edgecolor="#188038", linewidth=1.5)
    arrow_props = dict(arrowstyle="->", color="#202124", lw=1.6, mutation_scale=14)

    # Layers top to bottom
    layers = [
        ("Input MRI Slice: 224 x 224 x 3", box_norm, 10.2),
        ("Initial Conv2d (32 filters, 3x3, stride=2) + BatchNorm + ReLU6\nOutput Shape: 112 x 112 x 32", box_conv, 9.1),
        ("Inverted Residual Bottleneck Block 1 (stride=1, expansion=1)\nOutput Shape: 112 x 112 x 16", box_res, 8.0),
        ("Inverted Residual Blocks 2-3 (stride=2, expansion=6)\nOutput Shape: 56 x 56 x 24", box_res, 6.9),
        ("Inverted Residual Blocks 4-6 (stride=2, expansion=6)\nOutput Shape: 28 x 28 x 32", box_res, 5.8),
        ("Inverted Residual Blocks 7-14 (stride=2, expansion=6)\nOutput Shape: 14 x 14 x 96", box_res, 4.7),
        ("Inverted Residual Blocks 15-17 (stride=1/2, expansion=6)\nOutput Shape: 7 x 7 x 320 -> Conv2d 1x1 -> 7 x 7 x 1280", box_res, 3.6),
        ("Adaptive Average Pooling 2D (1 x 1) -> Flatten (1280-D)\n+ Dropout (p=0.3)", box_conv, 2.5),
        ("Dense Layer (1280 -> 256) + BatchNorm1d + ReLU + Dropout (p=0.2)", box_dense, 1.4),
        ("Linear Output Classification Head (256 -> 4 Classes)\n+ Softmax Diagnostic Activation", box_norm, 0.4)
    ]

    for i in range(len(layers)):
        text, style, y = layers[i]
        ax.text(5, y, text, ha="center", va="center", bbox=style, fontsize=8.8)
        if i < len(layers) - 1:
            next_y = layers[i+1][2]
            ax.annotate("", xy=(5, next_y + 0.4), xytext=(5, y - 0.4), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {save_path}")


def create_pipeline_diagram(save_path: str):
    """Generates Figure 3: Implementation pipeline diagram."""
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    arrow_props = dict(arrowstyle="->", color="#3c4043", lw=2.0, mutation_scale=16)

    stages = [
        ("Raw Brain MRI\nImage Input\n(DICOM / JPG)", "#e8eaed", "#5f6368", 1.2),
        ("Preprocessing &\nAugmentation\n(CLAHE + Resizing)", "#fedfc8", "#b06000", 3.7),
        ("Deep Transfer\nLearning Model\n(MobileNetV2)", "#d2e3fc", "#185abc", 6.2),
        ("Multi-Class Tumor\nClassification &\nRisk Report", "#ceead6", "#137333", 8.7)
    ]

    for name, fc, ec, x in stages:
        box = dict(boxstyle="round,pad=0.6", facecolor=fc, edgecolor=ec, linewidth=1.8)
        ax.text(x, 2.0, name, ha="center", va="center", bbox=box, fontsize=9.5, fontweight="bold")

    ax.annotate("", xy=(2.7, 2.0), xytext=(2.1, 2.0), arrowprops=arrow_props)
    ax.annotate("", xy=(5.2, 2.0), xytext=(4.6, 2.0), arrowprops=arrow_props)
    ax.annotate("", xy=(7.7, 2.0), xytext=(7.1, 2.0), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved Figure 3 to {save_path}")


if __name__ == "__main__":
    create_system_flowchart("assets/figure1_system_flowchart.png")
    create_architecture_diagram("assets/figure2_model_architecture.png")
    create_pipeline_diagram("assets/figure3_implementation_pipeline.png")
