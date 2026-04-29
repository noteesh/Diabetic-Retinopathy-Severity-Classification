import torch.nn as nn
from torchvision import models

class ImprovedDRClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3, freeze_backbone=False):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        
        for param in self.model.features.parameters():
            param.requires_grad = not freeze_backbone

    def forward(self, x):
        return self.model(x)

    def get_last_conv_layer(self):
        return self.model.features[-1]
