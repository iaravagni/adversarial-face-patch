# scripts/defense/02_test_detector.py

import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import os
import sys

# Ensure src is in the path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import from structured files
from src.data.dataset import load_lfw_dataset, LFWTensorDataset, PatchDataset
from src.models.patch_detection_model import PatchDetector
from src.defense.patch_detection import generate_patch_bank, get_predictions
from src.utils.visualization import plot_confusion_matrix, visualize_predictions
from src.utils.metrics import calculate_and_report_metrics
from src.utils.config import load_config

# ============================================================
# CONFIGURATION & SETUP
# ============================================================
# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
CONFIG = load_config(CONFIG_PATH)
if not CONFIG:
    sys.exit("Failed to load configuration. Exiting.")

# --- ROBUST PATH RESOLUTION ---

# 1. Determine the absolute path of the 'backend' root directory.
# This works reliably since the script is always inside backend/scripts/defense/
BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) 

# 2. Get the model directory path from the config (e.g., 'models')
MODEL_DIR_RELATIVE = CONFIG.get('models_dir', 'models') 

# 3. Construct the absolute directory path where the model is saved
MODEL_DIR_ABSOLUTE = os.path.join(BACKEND_ROOT, MODEL_DIR_RELATIVE)

# 4. Get the specific filename from the config
MODEL_FILENAME = CONFIG.get('patch_detector_filename', 'patch_detector.pth')

# 5. Construct the final absolute path for loading the file (SAVE_PATH)
SAVE_PATH = os.path.join(MODEL_DIR_ABSOLUTE, MODEL_FILENAME)

# Logging the path for verification
print(f"Loading model from absolute path: {SAVE_PATH}")


# Test Hyperparameters (must match training setup)
NUM_PATCHES_BANK = 100
TEST_SAMPLES = 200
BATCH_SIZE = 16
MAX_LFW_IMAGES = 1000 # Max images used from LFW for the base dataset

# Set device
DEVICE = torch.device(CONFIG.get('device', 'cpu') if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================
# STEP 1: LOAD BASE DATASET AND PATCHES
# ============================================================
print("\nLoading LFW dataset...")
# Load raw LFW data
imgs_raw, targets_raw, _ = load_lfw_dataset(min_faces_per_person=70) 

# Apply subset selection to match training setup
if imgs_raw.shape[0] > MAX_LFW_IMAGES:
    indices = np.random.choice(imgs_raw.shape[0], MAX_LFW_IMAGES, replace=False)
    imgs_raw = imgs_raw[indices]
    targets_raw = targets_raw[indices]

# Create the PyTorch-ready base dataset
lfw_base_dataset = LFWTensorDataset(imgs_raw, targets_raw, target_size=(128, 128))

# Generate patch bank (must use same seed/logic as training to generate the same test distribution)
patches, masks = generate_patch_bank(NUM_PATCHES_BANK)

# ============================================================
# STEP 2: CREATE TEST DATASET
# ============================================================
print("\nCreating testing Patch Dataset...")
test_data = PatchDataset(lfw_base_dataset, patches, masks, TEST_SAMPLES)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

print(f'Test Dataset size: {len(test_data)} samples')

# ============================================================
# STEP 3: LOAD MODEL
# ============================================================
print("\nLoading Patch Detector model...")
model = PatchDetector().to(DEVICE)

if not os.path.exists(SAVE_PATH):
    sys.exit(f"Error: Model weights not found at {SAVE_PATH}. Please run 01_train_detector.py first.")

model.load_state_dict(torch.load(SAVE_PATH))
model.eval()
print("✓ Model loaded successfully.")

# ============================================================
# STEP 4: EVALUATION
# ============================================================
print("\n--- Running Final Evaluation ---")

# Get all predictions
all_labels, all_preds = get_predictions(model, test_loader, DEVICE)

# Report metrics
target_names_detector = ['Clean', 'Patch']
calculate_and_report_metrics(all_labels, all_preds, target_names=target_names_detector)

# Visualize results
plot_confusion_matrix(all_labels, all_preds, classes=target_names_detector)
visualize_predictions(model, test_data, DEVICE)

print("\n✅ Patch Detector Test Complete!")