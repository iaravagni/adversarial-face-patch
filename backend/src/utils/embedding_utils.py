"""
Utility functions for working with face embeddings.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def load_all_embeddings(embeddings_path="data/processed/embeddings/all_embeddings.pt"):
    """
    Load all employee embeddings from file.
    
    Returns:
        dict: {'embeddings': {employee_id: tensor}, 'threshold': float, 'employee_ids': list}
    """
    return torch.load(embeddings_path)


def load_single_embedding(employee_id, embeddings_dir="data/processed/embeddings"):
    """
    Load a single employee's embedding.
    
    Args:
        employee_id: Employee identifier
        embeddings_dir: Directory containing embeddings
    
    Returns:
        dict: {'employee_id': str, 'embedding': tensor, 'threshold': float}
    """
    emb_path = Path(embeddings_dir) / f"{employee_id}_embedding.pt"
    return torch.load(emb_path)


def compute_similarity_matrix(embeddings_dict):
    """
    Compute pairwise cosine similarity matrix between all embeddings.
    
    Args:
        embeddings_dict: Dict of {employee_id: embedding_tensor}
    
    Returns:
        numpy array: Similarity matrix, list of employee IDs
    """
    employee_ids = list(embeddings_dict.keys())
    n = len(employee_ids)
    
    similarity_matrix = np.zeros((n, n))
    
    for i, emp1 in enumerate(employee_ids):
        for j, emp2 in enumerate(employee_ids):
            emb1 = embeddings_dict[emp1]
            emb2 = embeddings_dict[emp2]
            
            sim = torch.nn.functional.cosine_similarity(
                emb1.unsqueeze(0), 
                emb2.unsqueeze(0)
            ).item()
            
            similarity_matrix[i, j] = sim
    
    return similarity_matrix, employee_ids


def visualize_similarity_matrix(similarity_matrix, employee_ids, output_path=None):
    """
    Visualize similarity matrix as heatmap.
    
    Args:
        similarity_matrix: numpy array of similarities
        employee_ids: List of employee identifiers
        output_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # Shorten labels for display
    short_labels = [eid.split('_')[-1][:15] for eid in employee_ids]
    
    plt.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.xticks(range(len(short_labels)), short_labels, rotation=45, ha='right')
    plt.yticks(range(len(short_labels)), short_labels)
    plt.title('Employee Embedding Similarity Matrix', fontsize=14, fontweight='bold')
    
    # Add values to cells
    for i in range(len(employee_ids)):
        for j in range(len(employee_ids)):
            plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                    ha='center', va='center', 
                    color='white' if similarity_matrix[i, j] < 0.5 else 'black',
                    fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Similarity matrix saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_embeddings_2d(embeddings_dict, method='tsne', output_path=None):
    """
    Visualize embeddings in 2D using dimensionality reduction.
    
    Args:
        embeddings_dict: Dict of {employee_id: embedding_tensor}
        method: 'tsne' or 'pca'
        output_path: Path to save figure (optional)
    """
    employee_ids = list(embeddings_dict.keys())
    embeddings = torch.stack([embeddings_dict[eid] for eid in employee_ids]).numpy()
    
    # Dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(employee_ids)-1))
        title = 't-SNE Visualization of Employee Embeddings'
    else:
        reducer = PCA(n_components=2, random_state=42)
        title = 'PCA Visualization of Employee Embeddings'
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.6, c=range(len(employee_ids)), cmap='tab10')
    
    # Add labels
    for i, eid in enumerate(employee_ids):
        short_label = eid.split('_')[-1][:15]
        plt.annotate(short_label, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                    fontsize=9, alpha=0.8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Embedding visualization saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def get_embedding_statistics(embeddings_dict):
    """
    Compute statistics about embeddings.
    
    Args:
        embeddings_dict: Dict of {employee_id: embedding_tensor}
    
    Returns:
        dict: Statistics about embeddings
    """
    embeddings = torch.stack(list(embeddings_dict.values()))
    
    stats = {
        'num_employees': len(embeddings_dict),
        'embedding_dim': embeddings.shape[1],
        'mean_norm': torch.norm(embeddings, dim=1).mean().item(),
        'std_norm': torch.norm(embeddings, dim=1).std().item(),
        'mean_values': embeddings.mean(dim=0).numpy(),
        'std_values': embeddings.std(dim=0).numpy()
    }
    
    return stats


def find_most_similar_employees(embeddings_dict, top_k=3):
    """
    Find most similar employee pairs.
    
    Args:
        embeddings_dict: Dict of {employee_id: embedding_tensor}
        top_k: Number of top similar pairs to return
    
    Returns:
        list: [(emp1, emp2, similarity), ...]
    """
    similarity_matrix, employee_ids = compute_similarity_matrix(embeddings_dict)
    
    # Get upper triangle (excluding diagonal)
    n = len(employee_ids)
    similarities = []
    
    for i in range(n):
        for j in range(i+1, n):
            similarities.append((
                employee_ids[i],
                employee_ids[j],
                similarity_matrix[i, j]
            ))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    return similarities[:top_k]


def find_least_similar_employees(embeddings_dict, top_k=3):
    """
    Find least similar employee pairs.
    
    Args:
        embeddings_dict: Dict of {employee_id: embedding_tensor}
        top_k: Number of top dissimilar pairs to return
    
    Returns:
        list: [(emp1, emp2, similarity), ...]
    """
    similarity_matrix, employee_ids = compute_similarity_matrix(embeddings_dict)
    
    # Get upper triangle (excluding diagonal)
    n = len(employee_ids)
    similarities = []
    
    for i in range(n):
        for j in range(i+1, n):
            similarities.append((
                employee_ids[i],
                employee_ids[j],
                similarity_matrix[i, j]
            ))
    
    # Sort by similarity (ascending)
    similarities.sort(key=lambda x: x[2])
    
    return similarities[:top_k]


# Example usage
if __name__ == "__main__":
    print("Loading embeddings...")
    data = load_all_embeddings()
    embeddings = data['embeddings']
    
    print(f"\nLoaded {len(embeddings)} employee embeddings")
    
    # Statistics
    stats = get_embedding_statistics(embeddings)
    print(f"\nEmbedding Statistics:")
    print(f"  Dimension: {stats['embedding_dim']}")
    print(f"  Mean norm: {stats['mean_norm']:.3f}")
    print(f"  Std norm: {stats['std_norm']:.3f}")
    
    # Most similar
    print(f"\nMost Similar Employees:")
    for emp1, emp2, sim in find_most_similar_employees(embeddings):
        print(f"  {emp1} <-> {emp2}: {sim:.3f}")
    
    # Least similar
    print(f"\nLeast Similar Employees:")
    for emp1, emp2, sim in find_least_similar_employees(embeddings):
        print(f"  {emp1} <-> {emp2}: {sim:.3f}")
    
    # Visualize
    output_dir = Path("results/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    similarity_matrix, employee_ids = compute_similarity_matrix(embeddings)
    visualize_similarity_matrix(similarity_matrix, employee_ids, 
                               output_dir / "similarity_matrix.png")
    
    visualize_embeddings_2d(embeddings, method='tsne',
                           output_path=output_dir / "embeddings_tsne.png")
    
    visualize_embeddings_2d(embeddings, method='pca',
                           output_path=output_dir / "embeddings_pca.png")
    
    print(f"\nVisualizations saved to: {output_dir}")