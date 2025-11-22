"""
Face recognition model using FaceNet (InceptionResnetV1).
Handles face detection, embedding extraction, and classification.
"""

import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Dict
import pickle


class FaceRecognitionModel:
    """Face recognition model for employee identification"""
    
    def __init__(self, device: torch.device):
        """
        Initialize face recognition model.
        
        Args:
            device: torch device (cuda or cpu)
        """
        self.device = device
        
        # Initialize MTCNN for face detection
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            device=device,
            keep_all=False
        )
        
        # Initialize InceptionResnetV1 for face recognition
        self.model = InceptionResnetV1(
            pretrained='vggface2'
        ).eval().to(device)
        
        # Employee database (embeddings)
        self.employee_embeddings: Dict[str, torch.Tensor] = {}
        
        print(f"Face recognition model initialized on {device}")
    
    def get_face_embedding(self, image: Image.Image) -> Optional[torch.Tensor]:
        """
        Extract face embedding from PIL Image.
        
        Args:
            image: PIL Image containing a face
            
        Returns:
            Face embedding tensor or None if no face detected
        """
        try:
            # Detect and crop face
            face = self.mtcnn(image)
            
            if face is None:
                return None
            
            # Get embedding
            face = face.unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(face)
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return None
    
    def cosine_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        return torch.nn.functional.cosine_similarity(emb1, emb2).item()
    
    def classify_face(
        self,
        test_embedding: torch.Tensor,
        threshold: float = 0.6
    ) -> Tuple[str, float]:
        """
        Classify face as employee or unknown.
        
        Args:
            test_embedding: Embedding to classify
            threshold: Similarity threshold for positive identification
            
        Returns:
            Tuple of (identified_name, confidence_score)
        """
        max_similarity = -1
        identified_as = "Unknown"
        
        for name, emb in self.employee_embeddings.items():
            similarity = self.cosine_similarity(test_embedding, emb)
            
            if similarity > max_similarity:
                max_similarity = similarity
                if similarity > threshold:
                    identified_as = name
        
        return identified_as, max_similarity
    
    def load_employee_database(self, db_path: str):
        """
        Load employee embeddings database.
        
        Args:
            db_path: Path to pickled employee embeddings
        """
        try:
            with open(db_path, 'rb') as f:
                self.employee_embeddings = pickle.load(f)
            
            print(f"✓ Loaded {len(self.employee_embeddings)} employees")
            
        except Exception as e:
            print(f"Error loading employee database: {e}")
    
    def save_employee_database(self, db_path: str):
        """
        Save employee embeddings database.
        
        Args:
            db_path: Path to save pickled employee embeddings
        """
        try:
            with open(db_path, 'wb') as f:
                pickle.dump(self.employee_embeddings, f)
            
            print(f"✓ Saved {len(self.employee_embeddings)} employees")
            
        except Exception as e:
            print(f"Error saving employee database: {e}")
    
    def add_employee(self, name: str, images: list):
        """
        Add employee to database by averaging embeddings from multiple images.
        
        Args:
            name: Employee name/ID
            images: List of PIL Images of the employee
        """
        embeddings = []
        
        for img in images:
            emb = self.get_face_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        
        if embeddings:
            # Average embeddings
            avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
            self.employee_embeddings[name] = avg_embedding
            print(f"✓ Added {name} with {len(embeddings)} images")
        else:
            print(f"✗ Failed to add {name} - no valid embeddings")