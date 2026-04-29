import torch.nn as nn
from torchvision import models


class ImprovedDRClassifier(nn.Module):
    """
    EfficientNet-B4 fine-tuned for 5-class DR severity classification.
    Pretrained on ImageNet; classifier head replaced with a lightweight MLP.
    """

    def __init__(self, num_classes=5, dropout=0.3, freeze_backbone=False):
        super().__init__()
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b4(weights=weights)

        in_features = self.model.classifier[1].in_features  # 1792
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        self.freeze_features(freeze_backbone)

    def forward(self, x):
        return self.model(x)

    def freeze_features(self, freeze=True):
        """Optionally train only the classifier head for quicker first-pass results."""
        for param in self.model.features.parameters():
            param.requires_grad = not freeze

    def get_last_conv_layer(self):
        """Return last conv block for Grad-CAM hooks."""
        return self.model.features[-1]
