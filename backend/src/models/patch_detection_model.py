import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset

# ==========================================
# 1. The Model Architecture
# ==========================================
class PatchDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Architecture from Cell 5 of the notebook
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ==========================================
# 2. Patch Generation Helpers
# ==========================================
def create_circular_mask(size):
    """Create a circular mask for patch generation"""
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    mask = dist_from_center <= (size // 2)
    return torch.from_numpy(mask).float()

def generate_synthetic_patch(size=50, pattern='random'):
    """Generate a synthetic patch (Random, Noise, Gradient, Checkerboard)"""
    mask = create_circular_mask(size)

    if pattern == 'random':
        patch = torch.rand(3, size, size)
    elif pattern == 'noise':
        p = torch.randn(3, size, size) * 0.5 + 0.5
        patch = torch.clamp(p, 0, 1)
    elif pattern == 'gradient':
        x = torch.linspace(0, 1, size)
        y = torch.linspace(0, 1, size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        patch = torch.stack([xx, yy, xx*yy])
    else:  # checkerboard
        checker = torch.zeros(size, size)
        checker[::2, ::2] = 1
        checker[1::2, 1::2] = 1
        patch = checker.unsqueeze(0).repeat(3, 1, 1)

    # Apply circular mask
    for c in range(3):
        patch[c] = patch[c] * mask + (1 - mask) 
        
    return patch, mask

# ==========================================
# 3. Training Dataset Wrapper
# ==========================================
class PatchTrainingDataset(Dataset):
    """
    Wraps a standard dataset and dynamically applies patches 
    to 50% of images for robust training.
    """
    def __init__(self, base_dataset, num_samples=5000):
        self.base_dataset = base_dataset
        self.num_samples = num_samples
        
        # Pre-generate a pool of patches to speed up training
        self.patches = []
        self.masks = []
        print("Generating synthetic patches for training...")
        for pattern in ['random', 'noise', 'gradient', 'checkerboard']:
            for _ in range(25):
                size = random.randint(40, 80)
                patch, mask = generate_synthetic_patch(size, pattern)
                self.patches.append(patch)
                self.masks.append(mask)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Pick a random image from the base dataset
        base_idx = random.randint(0, len(self.base_dataset) - 1)
        image = self.base_dataset[base_idx] # Expecting Tensor (3, H, W)

        # Ensure image is a tensor
        if not isinstance(image, torch.Tensor):
            # Handle tuple (img, label) if base dataset returns it
            if isinstance(image, tuple):
                image = image[0]
            
        # 50% chance to add a patch
        if random.random() < 0.5:
            patch_idx = random.randint(0, len(self.patches) - 1)
            patch = self.patches[patch_idx]
            mask = self.masks[patch_idx]

            img = image.clone()
            _, img_h, img_w = img.shape
            _, patch_h, patch_w = patch.shape

            # Ensure patch fits
            if patch_w >= img_w or patch_h >= img_h:
                 # If patch is too big for image, return clean
                 return image, 0

            x = random.randint(0, img_w - patch_w)
            y = random.randint(0, img_h - patch_h)

            # Apply circular patch
            for c in range(3):
                img[c, y:y+patch_h, x:x+patch_w] = (
                    patch[c] * mask +
                    img[c, y:y+patch_h, x:x+patch_w] * (1 - mask)
                )

            return img, 1  # Label 1: Has Patch
        else:
            return image, 0  # Label 0: Clean