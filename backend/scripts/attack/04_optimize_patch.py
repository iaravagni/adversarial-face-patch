"""
Optimize adversarial patch to fool face recognition system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from facenet_pytorch import MTCNN

from src.models.face_recognition import FaceRecognitionModel
from src.attack.patch_generator import AdversarialPatchGenerator
from src.attack.patch_application import save_patch
from src.data.dataset import load_lfw_dataset, get_images_for_person
from src.utils.config import load_config, get_device


def main():
    """Optimize adversarial patch"""
    
    print("="*70)
    print("ADVERSARIAL PATCH OPTIMIZATION")
    print("="*70)
    
    # Load config
    config = load_config()
    device = get_device()
    
    # Load face recognition model
    face_model = FaceRecognitionModel(device=device)
    face_model.load_employee_database(config['employee_db_path'])
    
    # Load metadata
    metadata_path = config['employee_db_path'].replace('.pkl', '_metadata.pkl')
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Load dataset
    images, targets, target_names = load_lfw_dataset(color=True, min_faces_per_person=20)
    
    # Get attacker images
    attacker_id = metadata['attacker_ids'][0]
    attacker_name = target_names[attacker_id]
    
    print(f"\nAttacker: {attacker_name}")
    
    attacker_images = get_images_for_person(attacker_id, images, targets)
    
    # Split into train/test
    num_train = config['dataset']['attacker_train_images']
    attacker_train = attacker_images[:num_train]
    
    print(f"Using {len(attacker_train)} images for optimization")
    
    # Detect faces and prepare batch
    mtcnn = MTCNN(image_size=160, margin=20, device=device, keep_all=False)
    
    attacker_faces = []
    for img in attacker_train:
        face = mtcnn(img)
        if face is not None:
            attacker_faces.append(face)
    
    if not attacker_faces:
        print("ERROR: No faces detected in attacker images!")
        return
    
    attacker_batch = torch.stack(attacker_faces).to(device)
    print(f"✓ Detected {len(attacker_faces)} faces")
    
    # Select target employee (first one)
    target_employee = "Employee_1"
    target_embedding = face_model.employee_embeddings[target_employee]
    
    print(f"\nTarget: Impersonate {target_employee}")
    
    # Initialize patch generator
    patch_gen = AdversarialPatchGenerator(
        device=device,
        patch_radius=config['patch']['radius'],
        forehead_bounds=config['patch']['forehead_bounds']
    )
    
    # Optimize patch
    history = patch_gen.optimize_patch(
        attacker_images=attacker_batch,
        target_embedding=target_embedding,
        model=face_model.model,
        num_iterations=config['patch']['optimization']['iterations'],
        lr_content=config['patch']['optimization']['lr_content'],
        lr_position=config['patch']['optimization']['lr_position']
    )
    
    # Save patch
    patch_data = patch_gen.get_patch_data()
    patch_data['metadata']['target_employee'] = target_employee
    patch_data['metadata']['attacker_name'] = attacker_name
    patch_data['metadata']['success_rate'] = 0.0  # Will be updated after testing
    
    patch_name = f"patch_{target_employee.lower()}_r{config['patch']['radius']}"
    
    os.makedirs(config['patches_dir'], exist_ok=True)
    save_patch(
        patch_data['patch'],
        patch_data['metadata'],
        config['patches_dir'],
        patch_name
    )
    
    # Visualize optimization progress
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(history['losses'])
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Optimization Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['similarities'])
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Cosine Similarity')
    axes[0, 1].set_title(f'Similarity to {target_employee}')
    axes[0, 1].axhline(y=config['classification_threshold'], 
                       color='r', linestyle='--', label='Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(history['positions_x'], label='X')
    axes[1, 0].plot(history['positions_y'], label='Y')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Position (pixels)')
    axes[1, 0].set_title('Patch Position Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(history['positions_x'], history['positions_y'])
    axes[1, 1].scatter(history['positions_x'][0], history['positions_y'][0], 
                      c='green', s=100, marker='o', label='Start')
    axes[1, 1].scatter(history['positions_x'][-1], history['positions_y'][-1], 
                      c='red', s=100, marker='*', label='End')
    axes[1, 1].set_xlabel('X Position')
    axes[1, 1].set_ylabel('Y Position')
    axes[1, 1].set_title('Patch Movement Trajectory')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config['patches_dir']}/{patch_name}_optimization.png", dpi=150)
    print(f"✓ Saved optimization plot")
    
    # Visualize patch
    patch_vis = patch_data['patch'].permute(1, 2, 0).numpy()
    patch_vis = (patch_vis + 1) / 2
    patch_vis = np.clip(patch_vis, 0, 1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(patch_vis)
    plt.title(f'Optimized Adversarial Patch\nTarget: {target_employee}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{config['patches_dir']}/{patch_name}_visual.png", dpi=150)
    print(f"✓ Saved patch visualization")
    
    print("\n" + "="*70)
    print("✓ PATCH OPTIMIZATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()