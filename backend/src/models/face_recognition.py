"""
Face recognition model using FaceNet (InceptionResnetV1).
Handles face detection, embedding extraction, and classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
from typing import Optional, Tuple, Dict
import pickle
from src.utils.config import load_config


class FaceRecognitionModel:
    """Face recognition model for employee identification, using real names as keys."""
    
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
        
        # Employee database (embeddings) - keys will be real names (e.g., 'Bill Gates')
        self.employee_embeddings: Dict[str, torch.Tensor] = {}
        
        # Load the configuration to get the expected employee list for key remapping
        config = load_config()
        self.employee_names: List[str] = config['dataset']['specific_employees'] 
        
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
        return F.cosine_similarity(emb1, emb2).item()
    
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
            Tuple of (identified_name/Unknown, confidence_score) - uses real names.
        """
        max_similarity = -1
        identified_as = "Unknown"
        
        for name, emb in self.employee_embeddings.items():
            # 'name' is the real name (e.g., "Bill Gates")
            similarity = self.cosine_similarity(test_embedding, emb)
            
            if similarity > max_similarity:
                max_similarity = similarity
                if similarity > threshold:
                    identified_as = name 
        
        return identified_as, max_similarity
    
    def load_employee_database(self, db_path: str):
        """
        Load employee embeddings database and REMAP the keys from internal 
        labels (Employee_X) to real employee names (from the YAML config).
        
        Args:
            db_path: Path to pickled employee embeddings
        """
        try:
            with open(db_path, 'rb') as f:
                raw_db = pickle.load(f)
            
            remapped_db = {}
            num_employees = len(self.employee_names)
            
            for i in range(num_employees):
                internal_key = f"Employee_{i+1}"
                real_name = self.employee_names[i]
                
                if internal_key in raw_db:
                    remapped_db[real_name] = raw_db[internal_key]
                else:
                    print(f"Warning: Database missing expected key {internal_key} for {real_name}. Skipping.")
            
            self.employee_embeddings = remapped_db
            print(f"✓ Loaded and remapped {len(self.employee_embeddings)} employee embeddings (using real names as keys).")
            
        except Exception as e:
            print(f"Error loading employee embeddings from {db_path}: {e}")
            
    def save_employee_database(self, db_path: str):
        """
        Save employee embeddings database. NOTE: Saving will use the current real-name keys.
        
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
            name: Employee name/ID (real name)
            images: List of PIL Images of the employee
        """
        embeddings = []
        
        for img in images:
            emb = self.get_face_embedding(img)
            if emb is not None:
                embeddings.append(emb)
        
        if embeddings:
            avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
            self.employee_embeddings[name] = avg_embedding
            print(f"✓ Added {name} with {len(embeddings)} images")
        else:
            print(f"✗ Failed to add {name} - no valid embeddings")