"""
Configuration management utilities.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    # Default configuration
    default_config = {
        'employee_db_path': 'data/processed/embeddings/employee_db.pkl',
        'patches_dir': 'data/patches',
        'models_dir': 'models/pretrained',
        'classification_threshold': 0.55,
        'patch_detection_threshold': 0.15,
        'device': 'cuda',  # or 'cpu'
    }
    
    # Try to load from file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
                print(f"✓ Loaded config from {config_path}")
        except Exception as e:
            print(f"⚠ Error loading config: {e}")
            print("Using default configuration")
    else:
        print(f"⚠ Config file not found: {config_path}")
        print("Using default configuration")
    
    return default_config


def save_config(config: Dict[str, Any], config_path: str = "config.yaml"):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save config file
    """
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"✓ Saved config to {config_path}")
    except Exception as e:
        print(f"✗ Error saving config: {e}")


def get_device():
    """
    Get appropriate torch device (CUDA if available, else CPU).
    
    Returns:
        torch device
    """
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device