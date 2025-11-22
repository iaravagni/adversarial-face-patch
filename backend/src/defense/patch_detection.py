import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from typing import List, Tuple


# --- Patch Utility Functions ---
def create_circular_mask(size: int) -> torch.Tensor:
    """Create a circular mask (tensor) with radius size // 2."""
    center = size // 2
    y, x = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    mask = dist_from_center <= (size // 2)
    return torch.from_numpy(mask).float()

def generate_patch(size: int = 50, pattern: str = 'random') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a 3-channel circular patch of a specific pattern.
    The area outside the circle is set to white (1.0) using the mask.
    Returns: patch (3, size, size), mask (size, size)
    """
    mask = create_circular_mask(size)

    # Generate pattern based on type
    if pattern == 'random':
        patch = torch.rand(3, size, size)
    elif pattern == 'noise':
        p = torch.randn(3, size, size) * 0.5 + 0.5
        patch = torch.clamp(p, 0, 1)
    elif pattern == 'gradient':
        x = torch.linspace(0, 1, size)
        y = torch.linspace(0, 1, size)
        # Use torch.meshgrid with 'ij' indexing
        xx, yy = torch.meshgrid(x, y, indexing='ij') 
        patch = torch.stack([xx, yy, xx*yy])
    else:  # checkerboard
        checker = torch.zeros(size, size)
        checker[::2, ::2] = 1
        checker[1::2, 1::2] = 1
        patch = checker.unsqueeze(0).repeat(3, 1, 1)

    # Apply circular mask (make non-circle areas white)
    final_patch = torch.zeros_like(patch)
    for c in range(3):
        final_patch[c] = patch[c] * mask + (1 - mask)

    return final_patch, mask

def generate_patch_bank(num_patches: int = 100) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Generates a bank of diverse circular patches and their corresponding masks."""
    patches = []
    masks = []
    patterns = ['random', 'noise', 'gradient', 'checkerboard']
    num_per_pattern = num_patches // len(patterns)

    for pattern in patterns:
        for _ in range(num_per_pattern):
            size = random.randint(20, 50)
            patch, mask = generate_patch(size, pattern)
            patches.append(patch)
            masks.append(mask)

    print(f'Generated {len(patches)} circular patches bank.')
    return patches, masks

# --- Training and Evaluation Functions ---
def train_detector(
    model: nn.Module, 
    train_loader: DataLoader, 
    test_loader: DataLoader, 
    criterion: nn.Module, 
    optimizer: optim.Optimizer, 
    device: torch.device, 
    epochs: int, 
    save_path: str
):
    """Trains the patch detection model."""
    print("\n--- Training Patch Detector ---")
    best_acc = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, pred = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()

        # Evaluate on test set
        test_acc = evaluate_detector(model, test_loader, device) 

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)

        train_acc = 100 * train_correct / train_total
        print(f'Epoch {epoch:2d}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Best Acc: {best_acc:.2f}%')

    print(f'\n--- Training Complete ---')
    print(f'Best Accuracy Achieved: {best_acc:.2f}%')

def evaluate_detector(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Evaluates the patch detection model on a DataLoader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def get_predictions(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[List[int], List[int]]:
    """Gathers all predictions and true labels from a DataLoader."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_preds

