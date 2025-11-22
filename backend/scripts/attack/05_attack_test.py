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

    # For full determinism (slower, optional for CPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Test adversarial patch"""
    
    print("="*70)
    print("ADVERSARIAL PATCH TESTING")
    print("="*70)
    
    # Load config
    config = load_config()
    device = get_device()

    ATTACKER_INDEX = 0
    TARGET_INDEX = 0
    
    # Load face recognition model
    face_model = FaceRecognitionModel(device=device)
    face_model.load_employee_database(config["employee_db_path"])

    # Load metadata
    metadata_path = config["employee_db_path"].replace(".pkl", "_metadata.pkl")
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    raw_dir = config["raw_data_dir"]

    # Initialize MTCNN
    mtcnn = MTCNN(image_size=160, margin=20, device=device, keep_all=False)

    # Find patches
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

        print("\n" + "="*70)
        print(f"TESTING PATCH: {patch_file}")
        print("="*70)
        print(f"Original Patch Target: {patch_metadata['target_employee']}")
        print(f"Type: {patch_metadata['type']}")
        print(f"Size: {patch_metadata['size']} pixels")

        # ----------------------------------------------------
        # Select TARGET using TARGET_INDEX (same as optimize)
        # ----------------------------------------------------
        employee_labels = list(face_model.employee_embeddings.keys())
        target_employee = employee_labels[TARGET_INDEX]

        print(f"\nOverriding target → Using TARGET_INDEX={TARGET_INDEX}")
        print(f"Chosen Target Employee: {target_employee}")

        # ----------------------------------------------------
        # Select attacker with ATTACKER_INDEX (same as optimize)
        # ----------------------------------------------------
        attacker_id = metadata['attacker_ids'][ATTACKER_INDEX]
        attacker_name = metadata['target_names'][attacker_id]

        print(f"Attacker: {attacker_name}")

        # Load attacker images (SAME AS OPTIMIZE SCRIPT)
        attacker_images_list = load_saved_images(
            os.path.join(raw_dir, "attackers"),
            [attacker_name]
        )[0]

        # Test split (skip training images)
        num_train = config["dataset"]["attacker_train_images"]
        num_test = config["dataset"]["attacker_test_images"]
        test_images = attacker_images_list[num_train:num_train + num_test]

        print(f"Testing on {len(test_images)} images...")

        # Create mask for patch application
        if patch_metadata["type"] == "circular":
            radius = patch_metadata["size"] // 2
            circular_mask = create_circular_mask(
                patch_metadata["size"], radius, device
            )
        else:
            circular_mask = torch.ones(
                1, patch_metadata["size"], patch_metadata["size"]
            ).to(device)

        predictions = []
        confidences = []

        # Test each image
        for i, img in enumerate(test_images):
            face = mtcnn(img)

            if face is None:
                print(f"  Test {i+1}: [No face detected]")
                continue

            face = face.to(device)

            # Apply patch
            face_patched = apply_circular_patch(
                face,
                patch_tensor,
                patch_metadata["position"]["x"],
                patch_metadata["position"]["y"],
                circular_mask
            )
            face_patched = torch.clamp(face_patched, -1, 1)

            # Get prediction
            with torch.no_grad():
                embedding = face_model.model(face_patched.unsqueeze(0))

            identified, confidence = face_model.classify_face(
                embedding,
                threshold=config["classification_threshold"]
            )

            predictions.append(identified)
            confidences.append(confidence)

            status = "✓ SUCCESS" if identified == target_employee else "✗ FAILED"
            print(f"  Test {i+1}: {identified:15s} (conf: {confidence:.3f}) {status}")

        # Evaluate results
        results = evaluator.evaluate_targeted_attack(
            predictions,
            target=target_employee,
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
        patch_metadata["tested_target"] = target_employee
        patch_metadata["tested_attacker"] = attacker_name
        
        metadata_file = patch_path.replace(".pt", "_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(patch_metadata, f, indent=2)

        print(f"\n✓ Updated patch metadata: {metadata_file}")

    print("\n" + "="*70)
    print("✓ ATTACK TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    set_seed(42)
    main()