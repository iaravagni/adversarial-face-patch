"""
Download and prepare LFW dataset.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.dataset import load_lfw_dataset, select_people_for_experiment, save_selected_people
from src.utils.config import load_config



def main():
    """Download LFW dataset"""
    
    print("="*70)
    print("DOWNLOADING LFW DATASET")
    print("="*70)
    
    # Load config
    config = load_config()
    
    # Download dataset
    print("\nDownloading Labeled Faces in the Wild (LFW) dataset...")
    print("This may take a few minutes on first run...\n")
    
    images, targets, target_names = load_lfw_dataset(
        color=True,
        min_faces_per_person=20,
    )

    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    print(f"Total images: {len(images)}")
    print(f"Total people: {len(target_names)}")
    print(f"Image shape: {images[0].shape}")
    
    # Show distribution
    unique, counts = np.unique(targets, return_counts=True)
    
    print(f"\nImages per person:")
    print(f"  Min: {counts.min()}")
    print(f"  Max: {counts.max()}")
    print(f"  Mean: {counts.mean():.1f}")
    print(f"  Median: {np.median(counts):.1f}")
    
    print("\nSelecting employees and attackers...")
    employee_ids, attacker_ids = select_people_for_experiment(
        targets,
        target_names,
        num_employees=config["dataset"]["num_employees"],
        num_attackers=config["dataset"]["num_attackers"]
    )

    print("\nSaving selected images only...")
    save_selected_people(
        save_root=config["raw_data_dir"],
        images=images,
        targets=targets,
        target_names=target_names,
        employee_ids=employee_ids,
        attacker_ids=attacker_ids
    )

    print("\nSelected People (Employees + Attackers):")
    print("----------------------------------------")

    for pid in employee_ids:
        name = target_names[pid]
        count = np.sum(targets == pid)
        print(f"Employee – {name}: {count} images")

    for pid in attacker_ids:
        name = target_names[pid]
        count = np.sum(targets == pid)
        print(f"Attacker – {name}: {count} images")

    print("\n" + "="*70)
    print("✓ DATASET DOWNLOAD COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()