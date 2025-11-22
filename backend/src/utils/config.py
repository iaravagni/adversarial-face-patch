"""
Configuration management utilities.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    If no path is provided, load config.yaml from the backend directory.
    """
    # Determine the backend directory (where this utils folder lives)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    default_path = os.path.join(base_dir, "config.yaml")
    
    # If user didn't pass a path, use backend/config.yaml
    config_path = config_path or default_path
    
    default_config = {
        'employee_db_path': 'data/processed/embeddings/employee_db.pkl',
        'patches_dir': 'data/patches',
        'models_dir': 'models/pretrained',
        'classification_threshold': 0.55,
        'patch_detection_threshold': 0.15,
        'device': 'cuda',  # or 'cpu'
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
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