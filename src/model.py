"""
Baseline CNN model for Diabetic Retinopathy severity classification.

Intentionally simple architecture (3 conv blocks, no pretrained weights)
to serve as the performance baseline before applying more advanced techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Simple 3-block CNN baseline.

    Design choices that limit performance (intentionally for baseline):
    - No pretrained weights
    - Only 3 conv blocks → limited feature depth
    - No batch norm after fc layers
    - No dropout beyond a single layer
    """

    def __init__(self, num_classes=5, input_size=224):
        super(BaselineCNN, self).__init__()

        # Block 1: 3 -> 32 channels
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112
        )

        # Block 2: 32 -> 64 channels
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56
        )

        # Block 3: 64 -> 128 channels — this is the last conv layer, used for Grad-CAM
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28
        )

        # Global average pooling → reduces 128 x 28 x 28 to 128
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x

    def get_last_conv_layer(self):
        """Return last conv layer for Grad-CAM."""
        return self.block3[0]  # the Conv2d in block3
