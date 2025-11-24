"""
Test adversarial patch against face recognition system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from facenet_pytorch import MTCNN
import random
import json

from src.models.face_recognition import FaceRecognitionModel
from src.attack.patch_application import load_patch, apply_circular_patch, create_circular_mask
from src.data.dataset import load_saved_images
from src.utils.config import load_config, get_device
from src.attack.evaluator import AttackEvaluator


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Test adversarial patch"""
    
    print("="*70)
    print("ADVERSARIAL PATCH TESTING (REAL NAME LOGIC)")
    print("="*70)
    
    set_seed(42)
    config = load_config()
    device = get_device()

    # The FaceRecognitionModel MUST be the version where load_employee_database 
    # remaps the keys from 'Employee_X' to real names.
    face_model = FaceRecognitionModel(device=device)
    face_model.load_employee_database(config["employee_db_path"]) 

    mtcnn = MTCNN(image_size=160, margin=20, device=device, keep_all=False)
    
    patches_dir = config["patches_dir"]
    patch_files = [f for f in os.listdir(patches_dir) if f.endswith(".pt")]

    if not patch_files:
        print("ERROR: No patches found! Run 04_optimize_patch.py first.")
        return

    print(f"\nFound {len(patch_files)} patch(es)")
    evaluator = AttackEvaluator()

    for patch_file in patch_files:
        patch_path = os.path.join(patches_dir, patch_file)
        patch_tensor, patch_metadata = load_patch(patch_path)

        # Use the intended real name as the direct target for evaluation
        intended_target_real_name = patch_metadata.get('target_employee')
        attacker_name = patch_metadata.get('attacker', 'Unknown Attacker')

        # Skip if the intended target name is not in the remapped database keys
        if intended_target_real_name not in face_model.employee_embeddings:
            print(f"Skipping {patch_file}: Target employee '{intended_target_real_name}' not found in remapped database keys.")
            continue
            
        print("\n" + "="*70)
        print(f"TESTING PATCH: {patch_file}")
        print("="*70)
        print(f"Original Patch Target: {intended_target_real_name}")
        print(f"Type: {patch_metadata['type']}")
        print(f"Size: {patch_metadata['size']} pixels")
        print(f"Attacker: {attacker_name}")

        # Load attacker images
        attacker_images_list = load_saved_images(
            os.path.join(config["raw_data_dir"], "attackers"),
            [attacker_name]
        )[0]

        # Test split
        num_train = config["dataset"]["attacker_train_images"]
        num_test = config["dataset"]["attacker_test_images"]
        test_images = attacker_images_list[num_train:num_train + num_test]
        
        print(f"Testing on {len(test_images)} images...")

        # Create mask for patch application
        radius = patch_metadata["size"] // 2
        circular_mask = create_circular_mask(patch_metadata["size"], radius, device)

        predictions = []
        confidences = []

        # Assuming patch coordinates are stored in metadata
        patch_x = patch_metadata.get('position', {}).get('x', 80)
        patch_y = patch_metadata.get('position', {}).get('y', 30)

        # Test each image
        for i, img in enumerate(test_images):
            face = mtcnn(img)
            if face is None: continue
            face = face.to(device)
            
            face_patched = apply_circular_patch(face, patch_tensor, patch_x, patch_y, circular_mask)
            
            with torch.no_grad():
                embedding = face_model.model(face_patched.unsqueeze(0))
            
            # 'identified' is now the REAL NAME string
            identified, confidence = face_model.classify_face(embedding, threshold=config["classification_threshold"])

            predictions.append(identified)
            confidences.append(confidence)

            # Check success against the real name
            status = "✓ SUCCESS" if identified == intended_target_real_name else "✗ FAILED"
            # Print only the real name
            print(f"  Test {i+1}: {identified:15s} (conf: {confidence:.3f}) {status}")

        # Evaluate results
        results = evaluator.evaluate_targeted_attack(
            predictions,
            # Use the real name as the target for evaluation
            target=intended_target_real_name, 
            confidences=confidences
        )

        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Successes: {results['successes']}/{results['total_tests']}")
        print(f"Avg Confidence: {results['avg_confidence']:.3f}")

        # Update and save metadata
        patch_metadata["success_rate"] = results["success_rate"]
        # Save the real name to metadata
        patch_metadata["tested_target"] = intended_target_real_name
        patch_metadata["tested_attacker"] = attacker_name
        
        # Save the updated metadata
        metadata_file = patch_path.replace(".pt", "_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(patch_metadata, f, indent=2)

        print(f"\n✓ Updated patch metadata: {metadata_file}")

    print("\n" + "="*70)
    print("✓ ATTACK TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()