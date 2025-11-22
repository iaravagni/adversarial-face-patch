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
    num_attackers: int = 1
) -> Tuple[List[int], List[int]]:
    """
    Select people for employees and attackers based on image count.
    
    Args:
        targets: Array of target labels
        target_names: List of target names
        num_employees: Number of employees to select
        num_attackers: Number of attackers to select
        
    Returns:
        Tuple of (employee_ids, attacker_ids)
    """
    # Count images per person
    unique, counts = np.unique(targets, return_counts=True)
    sorted_idx = np.argsort(-counts)
    
    total_needed = num_employees + num_attackers
    selected_ids = unique[sorted_idx[:total_needed]]
    
    # Split into employees and attackers
    employee_ids = selected_ids[:num_employees].tolist()
    attacker_ids = selected_ids[num_employees:].tolist()
    
    print(f"\n✓ Selected {num_employees} employees and {num_attackers} attackers")
    
    for i, emp_id in enumerate(employee_ids):
        emp_name = target_names[emp_id]
        emp_count = counts[sorted_idx[i]]
        print(f"  Employee_{i+1}: {emp_name} ({emp_count} images)")
    
    for i, att_id in enumerate(attacker_ids):
        att_name = target_names[att_id]
        att_count = counts[sorted_idx[num_employees + i]]
        print(f"  Attacker_{i+1}: {att_name} ({att_count} images)")
    
    return employee_ids, attacker_ids