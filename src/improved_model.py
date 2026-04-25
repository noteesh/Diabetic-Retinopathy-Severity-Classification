import torch.nn as nn
from torchvision import models


class ImprovedDRClassifier(nn.Module):
    """
    EfficientNet-B0 fine-tuned for 5-class DR severity classification.
    Pretrained on ImageNet; classifier head replaced with a lightweight MLP.
    """

    def __init__(self, num_classes=5):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)

        in_features = self.model.classifier[1].in_features  # 1280
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.model(x)

    def get_last_conv_layer(self):
        """Return last conv block for Grad-CAM hooks."""
        return self.model.features[-1]
