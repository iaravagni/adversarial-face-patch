"""
Adversarial patch application module.
Handles loading and applying adversarial patches to images.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import Tuple, Optional
import json
import os


def load_patch(patch_path: str) -> Tuple[torch.Tensor, dict]:
    """
    Load adversarial patch and metadata.
    
    Args:
        patch_path: Path to .pt patch file
        
    Returns:
        Tuple of (patch_tensor, metadata_dict)
    """
    # Load patch
    patch_data = torch.load(patch_path, map_location='cpu')
    
    # Load metadata
    metadata_path = patch_path.replace('.pt', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return patch_data['patch'], metadata


def create_circular_mask(size: int, radius: int, device: torch.device) -> torch.Tensor:
    """
    Create circular mask for patch.
    
    Args:
        size: Size of the patch (width/height)
        radius: Radius of the circle
        device: torch device
        
    Returns:
        Binary mask tensor [1, H, W]
    """
    y_grid, x_grid = torch.meshgrid(
        torch.arange(size, dtype=torch.float32, device=device) - radius,
        torch.arange(size, dtype=torch.float32, device=device) - radius,
        indexing='ij'
    )
    circular_mask = (x_grid**2 + y_grid**2 <= radius**2).float()
    return circular_mask.unsqueeze(0)


def apply_circular_patch(
    face: torch.Tensor,
    patch: torch.Tensor,
    x_pos: float,
    y_pos: float,
    circular_mask: torch.Tensor
) -> torch.Tensor:
    """
    Apply circular patch to face image.
    
    Args:
        face: Face tensor [C, H, W] or [B, C, H, W]
        patch: Patch tensor [C, H, W]
        x_pos: X position of patch top-left corner
        y_pos: Y position of patch top-left corner
        circular_mask: Circular mask [1, H, W]
        
    Returns:
        Patched face tensor
    """
    # Handle both single image and batch
    is_batch = len(face.shape) == 4
    if not is_batch:
        face = face.unsqueeze(0)
    
    device = face.device
    batch_size = face.shape[0]
    face_size = face.shape[2]
    patch_h, patch_w = patch.shape[1], patch.shape[2]
    
    # Move patch and mask to device
    patch = patch.to(device)
    circular_mask = circular_mask.to(device)
    
    # Create coordinate grids
    y_face = torch.arange(face_size, dtype=torch.float32, device=device).view(-1, 1).expand(face_size, face_size)
    x_face = torch.arange(face_size, dtype=torch.float32, device=device).view(1, -1).expand(face_size, face_size)
    
    # Patch center in face coordinates
    patch_center_x = x_pos + patch_w / 2
    patch_center_y = y_pos + patch_h / 2
    
    # Translate to patch-centered coordinates
    x_centered = x_face - patch_center_x
    y_centered = y_face - patch_center_y
    
    # Map to patch coordinates
    x_patch = x_centered + patch_w / 2
    y_patch = y_centered + patch_h / 2
    
    # Normalize to [-1, 1] for grid_sample
    x_norm = (x_patch / patch_w) * 2 - 1
    y_norm = (y_patch / patch_h) * 2 - 1
    
    # Create sampling grid
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)
    
    # Sample from patch
    patch_batch = patch.unsqueeze(0).expand(batch_size, -1, -1, -1)
    sampled_patch = F.grid_sample(
        patch_batch, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    
    # Sample from mask
    mask_batch = circular_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)
    sampled_mask = F.grid_sample(
        mask_batch, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    
    # Smooth mask edges
    sampled_mask = torch.sigmoid((sampled_mask - 0.5) * 20)
    
    # Blend with face
    patched_face = face * (1 - sampled_mask) + sampled_patch * sampled_mask
    
    if not is_batch:
        patched_face = patched_face.squeeze(0)
    
    return patched_face


def apply_patch_to_image(image: Image.Image, patch_path: str) -> Image.Image:
    """
    Apply adversarial patch to PIL Image.
    
    Args:
        image: Input PIL Image
        patch_path: Path to patch .pt file
        
    Returns:
        Patched PIL Image
    """
    # Load patch and metadata
    patch_tensor, metadata = load_patch(patch_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert image to tensor
    from facenet_pytorch import MTCNN
    mtcnn = MTCNN(image_size=160, margin=20, device=device, keep_all=False)
    
    # Detect face
    face = mtcnn(image)
    if face is None:
        return image
    
    face = face.to(device)
    
    # Create circular mask
    patch_type = metadata.get('type', 'circular')
    if patch_type == 'circular':
        radius = metadata['size'] // 2
        circular_mask = create_circular_mask(
            metadata['size'],
            radius,
            device
        )
    else:
        # Square mask (all ones)
        circular_mask = torch.ones(1, metadata['size'], metadata['size']).to(device)
    
    # Apply patch
    patched_face = apply_circular_patch(
        face,
        patch_tensor,
        metadata['position']['x'],
        metadata['position']['y'],
        circular_mask
    )
    
    # Clamp values
    patched_face = torch.clamp(patched_face, -1, 1)
    
    # Convert back to PIL
    patched_np = patched_face.detach().cpu().permute(1, 2, 0).numpy()
    patched_np = ((patched_np + 1) / 2 * 255).astype(np.uint8)
    
    return Image.fromarray(patched_np)


def save_patch(
    patch: torch.Tensor,
    metadata: dict,
    save_dir: str,
    name: str
):
    """
    Save adversarial patch and metadata.
    
    Args:
        patch: Patch tensor
        metadata: Metadata dictionary
        save_dir: Directory to save files
        name: Patch name (without extension)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save patch tensor
    patch_path = os.path.join(save_dir, f"{name}.pt")
    torch.save({'patch': patch.cpu()}, patch_path)
    
    # Save metadata
    metadata_path = os.path.join(save_dir, f"{name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved patch to {patch_path}")
    print(f"✓ Saved metadata to {metadata_path}")