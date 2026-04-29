import torch.nn as nn
from torchvision import models

class ImprovedDRClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.4, freeze_backbone=False):
        super().__init__()
        # Load pre-trained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)
        
        # Replace the head
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        
        # Initial freezing state
        self.freeze_features(freeze_backbone)

    def forward(self, x):
        return self.model(x)

    def freeze_features(self, freeze=True):
        """
        Freezes or unfreezes the backbone (feature extractor) layers.
        Used for two-stage training.
        """
        for param in self.model.features.parameters():
            param.requires_grad = not freeze

    def get_last_conv_layer(self):
        """
        Returns the final convolutional block, useful for Grad-CAM 
        or other visualization techniques.
        """
        return self.model.features[-1]
