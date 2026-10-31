"""
NeuroScan-AI: Deep Learning Model Architecture Module
Provides custom Deep Residual CNN and Transfer Learning (MobileNetV2) architectures
for 4-class Brain Tumor MRI classification.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class NeuroScanCNN(nn.Module):
    """
    Custom 4-Stage Deep Convolutional Neural Network with Batch Normalization,
    Dropout regularization, and Global Average Pooling.
    """
    def __init__(self, num_classes: int = 4):
        super(NeuroScanCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Stage 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        pooled = self.global_pool(feat)
        out = self.classifier(pooled)
        return out


class NeuroScanTransfer(nn.Module):
    """
    Deep Transfer Learning Model utilizing MobileNetV2 with inverted residual blocks
    and linear bottlenecks, fine-tuned with a customized diagnostic classification head.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True, freeze_backbone: bool = False):
        super(NeuroScanTransfer, self).__init__()
        
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # MobileNetV2 feature extractor outputs 1280 channels
        in_features = self.backbone.classifier[1].in_features
        
        # Replace classifier with custom multi-layer medical diagnostic head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def build_model(model_type: str = "transfer", num_classes: int = 4, pretrained: bool = True):
    """Factory function to build and instantiate the chosen architecture."""
    if model_type.lower() == "transfer":
        model = NeuroScanTransfer(num_classes=num_classes, pretrained=pretrained)
    elif model_type.lower() == "custom_cnn":
        model = NeuroScanCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'transfer' or 'custom_cnn'.")
    return model


def count_parameters(model: nn.Module):
    """Computes total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    m_transfer = build_model("transfer")
    p_transfer = count_parameters(m_transfer)
    print(f"MobileNetV2 Transfer Model: {p_transfer['total']:,} total parameters ({p_transfer['trainable']:,} trainable).")
    
    m_cnn = build_model("custom_cnn")
    p_cnn = count_parameters(m_cnn)
    print(f"Custom CNN Baseline Model: {p_cnn['total']:,} total parameters ({p_cnn['trainable']:,} trainable).")
