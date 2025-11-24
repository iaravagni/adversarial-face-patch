"""
Dataset handling and loading utilities.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
from sklearn.datasets import fetch_lfw_people
from typing import List, Tuple, Optional


class FaceDataset(Dataset):
    """Dataset for face images"""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        transform=None
    ):
        """
        Initialize face dataset.
        
        Args:
            image_paths: List of image file paths
            labels: Optional list of labels
            transform: Optional transform to apply
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple:
        """Get image and label at index"""
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Return with or without label
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image
        

class LFWTensorDataset(Dataset):
    """
    A PyTorch Dataset for the LFW (Labeled Faces in the Wild) images.
    Handles conversion from sklearn's (N, H, W, 3) numpy array to
    PyTorch's (3, 128, 128) tensor format. This is used as the base
    for the PatchDataset when loading LFW raw data.
    """
    def __init__(self, images: np.ndarray, labels: np.ndarray, target_size: Tuple[int, int]=(128, 128)):
        # Pre-process and store images as tensors
        self.images = []
        self.labels = labels

        print(f"Resizing and converting base images to {target_size} tensor...")
        for img in images:
            # Resize
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = img_pil.resize(target_size)
            
            # Convert to tensor and normalize to [0,1]
            # Use torchvision.transforms for cleaner transformation if available, 
            # otherwise manually convert as in the original notebook:
            img_tensor = torch.from_numpy(np.array(img_pil)).float() / 255.0
            
            # Change from (H,W,3) to (3,H,W)
            img_tensor = img_tensor.permute(2, 0, 1)
            self.images.append(img_tensor)
        print("✓ Images converted.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# --- Patch Augmentation Dataset (from previous notebook) ---
class PatchDataset(Dataset):
    """
    Dataset that generates samples for patch detection training.
    It takes a base dataset (LFWTensorDataset) and randomly overlays 
    pre-generated patches 50% of the time.
    """
    def __init__(self, base_dataset: LFWTensorDataset, patches, masks, num_samples: int):
        self.base_dataset = base_dataset
        self.patches = patches
        self.masks = masks
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        base_idx = np.random.randint(0, len(self.base_dataset))
        image, _ = self.base_dataset[base_idx]

        # 50% chance to add a patch (label 1)
        if np.random.rand() < 0.5:
            patch_idx = np.random.randint(0, len(self.patches))
            patch = self.patches[patch_idx]
            mask = self.masks[patch_idx]

            img = image.clone()
            _, img_h, img_w = img.shape
            _, patch_h, patch_w = patch.shape

            # Choose random location for the patch
            x = np.random.randint(0, img_w - patch_w)
            y = np.random.randint(0, img_h - patch_h)

            # Apply circular patch with mask: New = Patch * Mask + Original * (1 - Mask)
            for c in range(3):
                img[c, y:y+patch_h, x:x+patch_w] = (
                    patch[c] * mask +
                    img[c, y:y+patch_h, x:x+patch_w] * (1 - mask)
                )

            return img, 1  # has patch
        else:
            # No patch (label 0)
            return image, 0  # clean

def load_lfw_dataset(
    color: bool = True,
    min_faces_per_person: int = 20,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load Labeled Faces in the Wild (LFW) dataset.
    
    Args:
        color: Whether to load color images
        min_faces_per_person: Minimum number of faces per person
        
    Returns:
        Tuple of (images, targets, target_names)
    """
    
    print("Loading LFW dataset...")
    lfw = fetch_lfw_people(
        color=color,
        resize=1.0,
        min_faces_per_person=min_faces_per_person
    )
    
    imgs = lfw.images
    targets = lfw.target
    target_names = lfw.target_names
    
    print(f"✓ Loaded {len(imgs)} images of {len(target_names)} people")
    
    return imgs, targets, target_names

def save_selected_people(
    save_root: str,
    images: np.ndarray,
    targets: np.ndarray,
    target_names: List[str],
    employee_ids: List[int],
    attacker_ids: List[int]
):
    """
    Save only selected employees and attackers into separate folders.

    Folder structure:
        save_root/employees/<person_name>/img_000.jpg
        save_root/attackers/<person_name>/img_000.jpg
    """
    emp_dir = os.path.join(save_root, "employees")
    att_dir = os.path.join(save_root, "attackers")

    os.makedirs(emp_dir, exist_ok=True)
    os.makedirs(att_dir, exist_ok=True)

    print(f"\nSaving selected people into {save_root}...\n")

    def save_group(ids, group_dir):
        for pid in ids:
            person_name = target_names[pid]
            person_dir = os.path.join(group_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)

            # get all indices for this person
            idxs = np.where(targets == pid)[0]

            for i, idx in enumerate(idxs):
                img = images[idx]

                # scale if needed
                if img.max() <= 1:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

                img_path = os.path.join(person_dir, f"img_{i:04d}.jpg")
                Image.fromarray(img).save(img_path)

            print(f"✓ Saved {len(idxs)} images of {person_name}")

    save_group(employee_ids, emp_dir)
    save_group(attacker_ids, att_dir)

    print("\n✓ Finished saving selected images!\n")

def load_saved_images(base_dir: str, person_names: List[str]) -> List[List[Image.Image]]:
    """
    Load saved face images from the raw folder.

    Args:
        base_dir: root directory containing employee/attacker folders
        person_names: list of folder names to load
    
    Returns:
        List of lists of PIL images
    """
    all_images = []

    for name in person_names:
        person_dir = os.path.join(base_dir, name)
        images = []

        for fname in sorted(os.listdir(person_dir)):
            if fname.lower().endswith((".jpg", ".png")):
                img = Image.open(os.path.join(person_dir, fname)).convert("RGB")
                images.append(img)

        all_images.append(images)

    return all_images


def get_images_for_person(
    person_id: int,
    images: np.ndarray,
    targets: np.ndarray
) -> List[Image.Image]:
    """
    Get all images for a specific person.
    
    Args:
        person_id: Person ID to get images for
        images: Array of all images
        targets: Array of all targets
        
    Returns:
        List of PIL Images for that person
    """
    indices = np.where(targets == person_id)[0]
    person_images = []
    
    for idx in indices:
        img_array = images[idx]
        
        # Scale to [0, 255] if needed
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        
        pil_img = Image.fromarray(img_array)
        person_images.append(pil_img)
    
    return person_images


def select_people_for_experiment(
    targets: np.ndarray,
    target_names: List[str],
    num_employees: int = 5,
    num_attackers: int = 1,
    specific_employees: Optional[List[str]] = None,
    specific_attackers: Optional[List[str]] = None
) -> Tuple[List[int], List[int]]:
    """
    Select people for employees and attackers. 
    Prioritizes specific names if provided; otherwise selects based on image count.
    """
    
    # Helper to find ID by name
    def get_id_by_name(name):
        try:
            # target_names is usually a numpy array or list
            return list(target_names).index(name)
        except ValueError:
            print(f"⚠️ Warning: '{name}' not found in dataset (or screened out by min_faces filter).")
            return None

    # --- 1. Select Employees ---
    employee_ids = []
    if specific_employees and len(specific_employees) > 0:
        print(f"\nLooking for specific employees: {specific_employees}")
        for name in specific_employees:
            pid = get_id_by_name(name)
            if pid is not None:
                employee_ids.append(pid)
    
    # --- 2. Select Attackers ---
    attacker_ids = []
    if specific_attackers and len(specific_attackers) > 0:
        print(f"Looking for specific attackers: {specific_attackers}")
        for name in specific_attackers:
            pid = get_id_by_name(name)
            if pid is not None:
                attacker_ids.append(pid)

    # --- 3. Fill remaining slots / Fallback logic ---
    # If we didn't specify names, or specified fewer than num_*, fill with top counts
    if len(employee_ids) < num_employees or len(attacker_ids) < num_attackers:
        
        # Calculate image counts for everyone
        unique, counts = np.unique(targets, return_counts=True)
        sorted_idx = np.argsort(-counts) # Indices of people with most images
        top_people_ids = unique[sorted_idx]
        
        # Fill Employees
        for pid in top_people_ids:
            if len(employee_ids) >= num_employees:
                break
            # Don't duplicate
            if pid not in employee_ids and pid not in attacker_ids:
                employee_ids.append(pid)
                
        # Fill Attackers
        for pid in top_people_ids:
            if len(attacker_ids) >= num_attackers:
                break
            # Don't duplicate and don't pick someone who is already an employee
            if pid not in attacker_ids and pid not in employee_ids:
                attacker_ids.append(pid)

    # Limit lists to exact requested number (in case specific list was too long)
    employee_ids = employee_ids[:num_employees]
    attacker_ids = attacker_ids[:num_attackers]

    print(f"\n✓ Final Selection: {len(employee_ids)} employees and {len(attacker_ids)} attackers")
    
    return employee_ids, attacker_ids