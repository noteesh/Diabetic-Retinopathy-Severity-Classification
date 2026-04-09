"""
Grad-CAM implementation for visualizing what the baseline CNN focuses on.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms


CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Hooks into the target conv layer to capture activations and gradients.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for given input.

        Args:
            input_tensor: (1, C, H, W) tensor
            class_idx: class to visualize; if None, uses predicted class

        Returns:
            cam: (H, W) numpy array with values in [0, 1]
            pred_class: predicted class index
        """
        self.model.eval()
        output = self.model(input_tensor)
        pred_class = output.argmax(dim=1).item()

        if class_idx is None:
            class_idx = pred_class

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)

        # Global average pool the gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, pred_class


def overlay_cam(original_img, cam, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        original_img: PIL Image or (H, W, 3) numpy array
        cam: (h, w) numpy array in [0, 1]
        alpha: blending factor for heatmap

    Returns:
        blended: (H, W, 3) numpy array
    """
    if isinstance(original_img, Image.Image):
        original_img = np.array(original_img.convert('RGB'))

    H, W = original_img.shape[:2]

    # Resize cam to original image size
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)) / 255.0

    # Colormap
    heatmap = cm.jet(cam_resized)[:, :, :3]  # (H, W, 3), drop alpha
    heatmap = (heatmap * 255).astype(np.uint8)

    blended = (alpha * heatmap + (1 - alpha) * original_img).astype(np.uint8)
    return blended


def visualize_gradcam(model, dataset, device, num_samples=8, seed=42):
    """
    Plot Grad-CAM overlays for a few samples from the dataset.

    Args:
        model: trained BaselineCNN
        dataset: APTOSDataset (with transform applied)
        device: torch device
        num_samples: how many images to show
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=num_samples, replace=False)

    # Get the last conv layer
    target_layer = model.get_last_conv_layer()
    gradcam = GradCAM(model, target_layer)

    # We also need the raw (un-normalized) image for display
    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    fig, axes = plt.subplots(num_samples, 2, figsize=(8, num_samples * 3))
    fig.suptitle('Grad-CAM Visualizations — Baseline CNN', fontsize=14, fontweight='bold')

    for row, idx in enumerate(indices):
        tensor, true_label = dataset[idx]
        input_tensor = tensor.unsqueeze(0).to(device)

        cam, pred_class = gradcam.generate(input_tensor)

        # Load original image for display
        img_path_stem = dataset.df.loc[idx, 'id_code']
        img_dir = dataset.img_dir
        for ext in ['.png', '.jpeg', '.jpg']:
            import os
            candidate = os.path.join(img_dir, img_path_stem + ext)
            if os.path.exists(candidate):
                raw_img = Image.open(candidate).convert('RGB')
                break
        raw_img_resized = raw_transform(raw_img)
        overlay = overlay_cam(raw_img_resized, cam)

        axes[row, 0].imshow(raw_img_resized)
        axes[row, 0].set_title(f'True: {CLASS_NAMES[true_label]}', fontsize=9)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(overlay)
        axes[row, 1].set_title(f'Pred: {CLASS_NAMES[pred_class]}', fontsize=9,
                               color='green' if pred_class == true_label else 'red')
        axes[row, 1].axis('off')

    plt.tight_layout()
    return fig
