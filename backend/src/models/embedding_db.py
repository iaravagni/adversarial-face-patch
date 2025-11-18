"""
Embedding database utilities for face recognition.
Manages storage and retrieval of face embeddings.
"""

import torch
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
import os


class EmbeddingDatabase:
    """Database for storing and querying face embeddings"""
    
    def __init__(self):
        """Initialize empty embedding database"""
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.metadata: Dict[str, dict] = {}
    
    def add_embedding(
        self,
        person_id: str,
        embedding: torch.Tensor,
        metadata: Optional[dict] = None
    ):
        """
        Add embedding to database.
        
        Args:
            person_id: Unique identifier for person
            embedding: Face embedding tensor
            metadata: Optional metadata dictionary
        """
        self.embeddings[person_id] = embedding.cpu()
        if metadata:
            self.metadata[person_id] = metadata
    
    def get_embedding(self, person_id: str) -> Optional[torch.Tensor]:
        """
        Retrieve embedding by person ID.
        
        Args:
            person_id: Person identifier
            
        Returns:
            Embedding tensor or None if not found
        """
        return self.embeddings.get(person_id)
    
    def remove_embedding(self, person_id: str):
        """
        Remove embedding from database.
        
        Args:
            person_id: Person identifier to remove
        """
        if person_id in self.embeddings:
            del self.embeddings[person_id]
        if person_id in self.metadata:
            del self.metadata[person_id]
    
    def search_similar(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """
        Search for most similar embeddings.
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            threshold: Optional similarity threshold
            
        Returns:
            List of (person_id, similarity_score) tuples
        """
        similarities = []
        
        for person_id, embedding in self.embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding,
                embedding
            ).item()
            
            if threshold is None or similarity >= threshold:
                similarities.append((person_id, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save(self, filepath: str):
        """
        Save database to file.
        
        Args:
            filepath: Path to save database
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Saved embedding database to {filepath}")
    
    def load(self, filepath: str):
        """
        Load database from file.
        
        Args:
            filepath: Path to load database from
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.embeddings = data['embeddings']
        self.metadata = data.get('metadata', {})
        
        print(f"✓ Loaded {len(self.embeddings)} embeddings from {filepath}")
    
    def get_statistics(self) -> dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        if not self.embeddings:
            return {
                'num_people': 0,
                'embedding_dim': 0,
                'total_size_mb': 0
            }
        
        # Get embedding dimension
        first_embedding = next(iter(self.embeddings.values()))
        embedding_dim = first_embedding.shape[-1]
        
        # Calculate approximate size
        total_elements = sum(e.numel() for e in self.embeddings.values())
        size_bytes = total_elements * 4  # float32 = 4 bytes
        size_mb = size_bytes / (1024 * 1024)
        
        return {
            'num_people': len(self.embeddings),
            'embedding_dim': embedding_dim,
            'total_size_mb': round(size_mb, 2)
        }
    
    def __len__(self) -> int:
        """Return number of embeddings in database"""
        return len(self.embeddings)
    
    def __contains__(self, person_id: str) -> bool:
        """Check if person_id exists in database"""
        return person_id in self.embeddings