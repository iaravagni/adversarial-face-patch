"""
Download and prepare LFW dataset.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.data.dataset import load_lfw_dataset
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
        min_faces_per_person=10,
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
    
    print(f"\nTop 10 people by image count:")
    sorted_idx = np.argsort(-counts)
    for i in range(min(10, len(sorted_idx))):
        person_idx = unique[sorted_idx[i]]
        person_name = target_names[person_idx]
        person_count = counts[sorted_idx[i]]
        print(f"  {i+1}. {person_name}: {person_count} images")
    

if __name__ == "__main__":
    main()