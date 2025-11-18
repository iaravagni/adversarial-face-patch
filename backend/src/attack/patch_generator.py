"""
Adversarial patch generation and optimization.
Creates patches that fool face recognition systems.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
import numpy as np


class AdversarialPatchGenerator:
    """Generator for adversarial patches targeting face recognition"""
    
    def __init__(
        self,
        device: torch.device,
        patch_radius: int = 25,
        forehead_bounds: Dict[str, int] = None
    ):
        """
        Initialize patch generator.
        
        Args:
            device: torch device
            patch_radius: Radius of circular patch
            forehead_bounds: Dict with x_min, x_max, y_min, y_max for patch positioning
        """
        self.device = device
        self.patch_radius = patch_radius
        self.patch_size = patch_radius * 2
        
        # Default forehead bounds (can be adjusted)
        if forehead_bounds is None:
            self.forehead_bounds = {
                'x_min': 0,
                'x_max': 160,
                'y_min': 0,
                'y_max': 160
            }
        else:
            self.forehead_bounds = forehead_bounds
        
        # Create circular mask
        self.circular_mask = self._create_circular_mask()
        
        # Initialize patch
        self.patch = None
        self.patch_x = None
        self.patch_y = None
    
    def _create_circular_mask(self) -> torch.Tensor:
        """Create circular mask for patch"""
        y_grid, x_grid = torch.meshgrid(
            torch.arange(self.patch_size, dtype=torch.float32, device=self.device) - self.patch_radius,
            torch.arange(self.patch_size, dtype=torch.float32, device=self.device) - self.patch_radius,
            indexing='ij'
        )
        circular_mask = (x_grid**2 + y_grid**2 <= self.patch_radius**2).float()
        return circular_mask.unsqueeze(0)
    
    def initialize_patch(self, init_x: float = 70.0, init_y: float = 10.0):
        """
        Initialize patch parameters.
        
        Args:
            init_x: Initial x position
            init_y: Initial y position
        """
        # Initialize patch content
        self.patch = torch.rand(
            3, self.patch_size, self.patch_size,
            requires_grad=True,
            device=self.device
        ) * 0.5
        self.patch = nn.Parameter(self.patch)
        
        # Initialize position
        self.patch_x = nn.Parameter(torch.tensor(init_x, device=self.device))
        self.patch_y = nn.Parameter(torch.tensor(init_y, device=self.device))
    
    def optimize_patch(
        self,
        attacker_images: torch.Tensor,
        target_embedding: torch.Tensor,
        model: nn.Module,
        num_iterations: int = 1000,
        lr_content: float = 0.2,
        lr_position: float = 3.0
    ) -> Dict:
        """
        Optimize patch to maximize similarity to target employee.
        
        Args:
            attacker_images: Batch of attacker face images [B, C, H, W]
            target_embedding: Target employee embedding to impersonate
            model: Face recognition model
            num_iterations: Number of optimization iterations
            lr_content: Learning rate for patch content
            lr_position: Learning rate for patch position
            
        Returns:
            Dictionary with optimization history
        """
        from .patch_application import apply_circular_patch
        
        # Initialize if not already done
        if self.patch is None:
            self.initialize_patch()
        
        # Setup optimizer
        optimizer = optim.Adam([
            {'params': [self.patch], 'lr': lr_content},
            {'params': [self.patch_x, self.patch_y], 'lr': lr_position}
        ])
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
        
        # Tracking
        history = {
            'losses': [],
            'similarities': [],
            'positions_x': [],
            'positions_y': []
        }
        
        best_similarity = -1
        best_state = None
        
        print(f"\nOptimizing patch for {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Constrain position to forehead
            x_pos = torch.clamp(
                self.patch_x,
                float(self.forehead_bounds['x_min']),
                float(self.forehead_bounds['x_max'])
            )
            y_pos = torch.clamp(
                self.patch_y,
                float(self.forehead_bounds['y_min']),
                float(self.forehead_bounds['y_max'])
            )
            
            # Apply patch to all attacker images
            patched_imgs = apply_circular_patch(
                attacker_images,
                self.patch,
                x_pos,
                y_pos,
                self.circular_mask
            )
            
            # Clamp to valid range
            patched_imgs = torch.clamp(patched_imgs, -1, 1)
            
            # Get embeddings
            embeddings = model(patched_imgs)
            
            # Calculate similarity to target
            avg_similarity = torch.nn.functional.cosine_similarity(
                embeddings,
                target_embedding.expand_as(embeddings)
            ).mean()
            
            # Track metrics
            history['similarities'].append(avg_similarity.item())
            history['positions_x'].append(x_pos.item())
            history['positions_y'].append(y_pos.item())
            
            # Update best
            if avg_similarity.item() > best_similarity:
                best_similarity = avg_similarity.item()
                best_state = {
                    'patch': self.patch.detach().clone(),
                    'x': x_pos.item(),
                    'y': y_pos.item()
                }
            
            # Loss: maximize similarity
            similarity_loss = -avg_similarity
            smoothness_loss = torch.var(self.patch) * 0.001
            loss = similarity_loss + smoothness_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Clamp patch values
            with torch.no_grad():
                self.patch.clamp_(-1, 1)
            
            history['losses'].append(loss.item())
            
            if (iteration + 1) % 50 == 0:
                print(f"  Iter {iteration + 1}/{num_iterations} | "
                      f"Sim: {avg_similarity.item():.3f} | "
                      f"Pos: ({x_pos.item():.1f}, {y_pos.item():.1f})")
        
        # Restore best state
        if best_state:
            self.patch.data = best_state['patch']
            self.patch_x.data = torch.tensor(best_state['x'], device=self.device)
            self.patch_y.data = torch.tensor(best_state['y'], device=self.device)
        
        print(f"\n✓ Optimization complete!")
        print(f"  Best similarity: {best_similarity:.3f}")
        print(f"  Final position: ({self.patch_x.item():.1f}, {self.patch_y.item():.1f})")
        
        return history
    
    def get_patch_data(self) -> Dict:
        """
        Get patch data for saving.
        
        Returns:
            Dictionary with patch tensor and metadata
        """
        return {
            'patch': self.patch.detach().cpu(),
            'metadata': {
                'type': 'circular',
                'size': self.patch_size,
                'radius': self.patch_radius,
                'position': {
                    'x': self.patch_x.item(),
                    'y': self.patch_y.item()
                },
                'bounds': self.forehead_bounds
            }
        }