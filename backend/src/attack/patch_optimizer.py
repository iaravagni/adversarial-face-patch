"""
Advanced patch optimization strategies.
Implements various optimization techniques for adversarial patches.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Callable, Optional
import numpy as np


class PatchOptimizer:
    """Advanced optimizer for adversarial patches"""
    
    def __init__(
        self,
        device: torch.device,
        optimization_strategy: str = 'adam'
    ):
        """
        Initialize patch optimizer.
        
        Args:
            device: torch device
            optimization_strategy: 'adam', 'sgd', or 'lbfgs'
        """
        self.device = device
        self.strategy = optimization_strategy
    
    def optimize_targeted(
        self,
        patch: nn.Parameter,
        position_params: list,
        loss_fn: Callable,
        num_iterations: int = 1000,
        lr_content: float = 0.2,
        lr_position: float = 3.0,
        scheduler_type: str = 'step'
    ) -> Dict:
        """
        Optimize patch with targeted attack objective.
        
        Args:
            patch: Patch parameter to optimize
            position_params: List of position parameters
            loss_fn: Loss function to minimize
            num_iterations: Number of optimization steps
            lr_content: Learning rate for patch content
            lr_position: Learning rate for position
            scheduler_type: Type of learning rate scheduler
            
        Returns:
            Dictionary with optimization history
        """
        # Setup optimizer based on strategy
        if self.strategy == 'adam':
            optimizer = optim.Adam([
                {'params': [patch], 'lr': lr_content},
                {'params': position_params, 'lr': lr_position}
            ])
        elif self.strategy == 'sgd':
            optimizer = optim.SGD([
                {'params': [patch], 'lr': lr_content, 'momentum': 0.9},
                {'params': position_params, 'lr': lr_position, 'momentum': 0.9}
            ])
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Setup scheduler
        if scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=100,
                gamma=0.7
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_iterations
            )
        else:
            scheduler = None
        
        # Tracking
        history = {
            'losses': [],
            'learning_rates': []
        }
        
        best_loss = float('inf')
        best_state = None
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Compute loss
            loss = loss_fn()
            
            # Track
            history['losses'].append(loss.item())
            history['learning_rates'].append(
                optimizer.param_groups[0]['lr']
            )
            
            # Update best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {
                    'patch': patch.detach().clone(),
                    'positions': [p.detach().clone() for p in position_params]
                }
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Clamp patch values
            with torch.no_grad():
                patch.clamp_(-1, 1)
        
        # Restore best state
        if best_state:
            patch.data = best_state['patch']
            for i, param in enumerate(position_params):
                param.data = best_state['positions'][i]
        
        return history
    
    def optimize_untargeted(
        self,
        patch: nn.Parameter,
        position_params: list,
        loss_fn: Callable,
        num_iterations: int = 1000,
        lr: float = 0.2
    ) -> Dict:
        """
        Optimize patch with untargeted attack objective.
        Goal: Maximize confusion in recognition system.
        
        Args:
            patch: Patch parameter
            position_params: Position parameters
            loss_fn: Loss function
            num_iterations: Number of steps
            lr: Learning rate
            
        Returns:
            Optimization history
        """
        optimizer = optim.Adam(
            [patch] + position_params,
            lr=lr
        )
        
        history = {'losses': []}
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            loss = loss_fn()
            
            history['losses'].append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                patch.clamp_(-1, 1)
        
        return history
    
    def optimize_with_constraints(
        self,
        patch: nn.Parameter,
        position_params: list,
        loss_fn: Callable,
        constraint_fn: Optional[Callable] = None,
        num_iterations: int = 1000,
        lr: float = 0.2,
        constraint_weight: float = 0.1
    ) -> Dict:
        """
        Optimize with additional constraints (e.g., smoothness, size).
        
        Args:
            patch: Patch parameter
            position_params: Position parameters
            loss_fn: Primary loss function
            constraint_fn: Optional constraint function
            num_iterations: Number of steps
            lr: Learning rate
            constraint_weight: Weight for constraint loss
            
        Returns:
            Optimization history
        """
        optimizer = optim.Adam(
            [patch] + position_params,
            lr=lr
        )
        
        history = {
            'losses': [],
            'constraint_losses': []
        }
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Primary loss
            loss = loss_fn()
            
            # Add constraints if provided
            if constraint_fn:
                constraint_loss = constraint_fn()
                total_loss = loss + constraint_weight * constraint_loss
                history['constraint_losses'].append(constraint_loss.item())
            else:
                total_loss = loss
            
            history['losses'].append(loss.item())
            
            total_loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                patch.clamp_(-1, 1)
        
        return history


class LossComposer:
    """Compose multiple loss functions"""
    
    @staticmethod
    def smoothness_loss(patch: torch.Tensor, weight: float = 0.001) -> torch.Tensor:
        """
        Encourage smooth patch patterns.
        
        Args:
            patch: Patch tensor [C, H, W]
            weight: Loss weight
            
        Returns:
            Smoothness loss
        """
        # Total variation loss
        diff_h = torch.abs(patch[:, 1:, :] - patch[:, :-1, :])
        diff_w = torch.abs(patch[:, :, 1:] - patch[:, :, :-1])
        
        tv_loss = diff_h.sum() + diff_w.sum()
        
        return weight * tv_loss
    
    @staticmethod
    def color_variance_loss(patch: torch.Tensor, weight: float = 0.001) -> torch.Tensor:
        """
        Penalize extreme color variance.
        
        Args:
            patch: Patch tensor [C, H, W]
            weight: Loss weight
            
        Returns:
            Color variance loss
        """
        variance = torch.var(patch)
        return weight * variance
    
    @staticmethod
    def norm_loss(patch: torch.Tensor, weight: float = 0.001) -> torch.Tensor:
        """
        Penalize large patch norms.
        
        Args:
            patch: Patch tensor
            weight: Loss weight
            
        Returns:
            Norm loss
        """
        return weight * torch.norm(patch)
    
    @staticmethod
    def printability_loss(
        patch: torch.Tensor,
        printable_colors: Optional[torch.Tensor] = None,
        weight: float = 0.1
    ) -> torch.Tensor:
        """
        Encourage printable colors (for physical patches).
        
        Args:
            patch: Patch tensor [C, H, W]
            printable_colors: Tensor of printable RGB colors [N, 3]
            weight: Loss weight
            
        Returns:
            Printability loss
        """
        if printable_colors is None:
            # Default set of printable colors
            printable_colors = torch.tensor([
                [-1, -1, -1],  # Black
                [1, 1, 1],      # White
                [1, -1, -1],    # Red
                [-1, 1, -1],    # Green
                [-1, -1, 1],    # Blue
                [1, 1, -1],     # Yellow
            ], device=patch.device, dtype=patch.dtype)
        
        # Reshape patch to [3, H*W]
        patch_flat = patch.reshape(3, -1).T  # [H*W, 3]
        
        # Compute distance to nearest printable color
        distances = torch.cdist(patch_flat, printable_colors)
        min_distances = distances.min(dim=1)[0]
        
        return weight * min_distances.mean()