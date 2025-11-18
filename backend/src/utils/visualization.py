"""
Visualization utilities for adversarial face recognition.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Optional, Tuple
import seaborn as sns


def plot_patch(
    patch: torch.Tensor,
    title: str = "Adversarial Patch",
    save_path: Optional[str] = None
):
    """
    Visualize adversarial patch.
    
    Args:
        patch: Patch tensor [C, H, W]
        title: Plot title
        save_path: Optional path to save figure
    """
    # Convert to numpy and normalize
    patch_np = patch.detach().cpu().permute(1, 2, 0).numpy()
    patch_np = (patch_np + 1) / 2  # [-1, 1] -> [0, 1]
    patch_np = np.clip(patch_np, 0, 1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(patch_np)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved patch visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_attack_comparison(
    original_image: torch.Tensor,
    patched_image: torch.Tensor,
    patch: torch.Tensor,
    original_pred: str,
    patched_pred: str,
    save_path: Optional[str] = None
):
    """
    Compare original and patched images side by side.
    
    Args:
        original_image: Original face tensor [C, H, W]
        patched_image: Patched face tensor [C, H, W]
        patch: Patch tensor [C, H, W]
        original_pred: Prediction on original
        patched_pred: Prediction on patched
        save_path: Optional save path
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert to displayable format
    def to_image(tensor):
        img = tensor.detach().cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        return np.clip(img, 0, 1)
    
    # Original
    axes[0].imshow(to_image(original_image))
    axes[0].set_title(f'Original\nPredicted: {original_pred}', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Patched
    axes[1].imshow(to_image(patched_image))
    axes[1].set_title(f'With Adversarial Patch\nPredicted: {patched_pred}', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Patch
    axes[2].imshow(to_image(patch))
    axes[2].set_title('Adversarial Patch', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_optimization_history(
    history: dict,
    save_path: Optional[str] = None
):
    """
    Plot optimization history (loss, similarity, position).
    
    Args:
        history: Dictionary with 'losses', 'similarities', 'positions_x', 'positions_y'
        save_path: Optional save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    if 'losses' in history:
        axes[0, 0].plot(history['losses'])
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Optimization Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Similarity
    if 'similarities' in history:
        axes[0, 1].plot(history['similarities'])
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Cosine Similarity')
        axes[0, 1].set_title('Similarity to Target')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Position over time
    if 'positions_x' in history and 'positions_y' in history:
        axes[1, 0].plot(history['positions_x'], label='X position')
        axes[1, 0].plot(history['positions_y'], label='Y position')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Position (pixels)')
        axes[1, 0].set_title('Patch Position Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 2D trajectory
        axes[1, 1].plot(history['positions_x'], history['positions_y'], 'b-', alpha=0.5)
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
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved optimization history to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(
    confusion: np.ndarray,
    classes: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.
    
    Args:
        confusion: Confusion matrix [n_classes, n_classes]
        classes: List of class names
        title: Plot title
        save_path: Optional save path
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        confusion,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_success_rates(
    baseline_rate: float,
    attack_rate: float,
    defense_rate: float,
    save_path: Optional[str] = None
):
    """
    Plot comparison of success rates.
    
    Args:
        baseline_rate: Baseline accuracy
        attack_rate: Attack success rate
        defense_rate: Defense success rate
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenarios = ['Baseline\n(No Attack)', 'Attack\n(No Defense)', 'Attack\n(With Defense)']
    rates = [baseline_rate, attack_rate, defense_rate]
    colors = ['green', 'red', 'blue']
    
    bars = ax.bar(scenarios, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Attack Success Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved success rate comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_detection_heatmap(
    image: torch.Tensor,
    detection_scores: torch.Tensor,
    title: str = "Patch Detection Heatmap",
    save_path: Optional[str] = None
):
    """
    Visualize patch detection scores as heatmap overlay.
    
    Args:
        image: Image tensor [C, H, W]
        detection_scores: Detection score map [H, W]
        title: Plot title
        save_path: Optional save path
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert image
    img_np = image.detach().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np + 1) / 2
    img_np = np.clip(img_np, 0, 1)
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Detection heatmap
    scores_np = detection_scores.detach().cpu().numpy()
    im = axes[1].imshow(scores_np, cmap='hot', interpolation='bilinear')
    axes[1].set_title('Detection Scores')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # Overlay
    axes[2].imshow(img_np)
    axes[2].imshow(scores_np, cmap='hot', alpha=0.5, interpolation='bilinear')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved detection heatmap to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_embedding_space(
    embeddings: dict,
    labels: List[str],
    title: str = "Face Embedding Space (t-SNE)",
    save_path: Optional[str] = None
):
    """
    Visualize face embeddings in 2D using t-SNE.
    
    Args:
        embeddings: Dictionary of embeddings
        labels: List of labels for each embedding
        title: Plot title
        save_path: Optional save path
    """
    from sklearn.manifold import TSNE
    
    # Stack embeddings
    embedding_list = []
    label_list = []
    
    for label, emb in embeddings.items():
        if isinstance(emb, torch.Tensor):
            emb = emb.detach().cpu().numpy()
        embedding_list.append(emb.flatten())
        label_list.append(label)
    
    X = np.array(embedding_list)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    unique_labels = list(set(label_list))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = [l == label for l in label_list]
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=[color], label=label, s=100, alpha=0.7, edgecolors='black')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved embedding space visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()