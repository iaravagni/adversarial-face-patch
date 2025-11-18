"""
Test adversarial patch attack effectiveness.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import pickle
import json
from facenet_pytorch import MTCNN

from src.models.face_recognition import FaceRecognitionModel
from src.attack.patch_application import load_patch, apply_circular_patch, create_circular_mask
from src.data.dataset import load_lfw_dataset, get_images_for_person
from src.utils.config import load_config, get_device
from src.attack.evaluator import AttackEvaluator


def main():
    """Test adversarial attack"""
    
    print("="*70)
    print("ADVERSARIAL ATTACK TEST")
    print("="*70)
    
    # Load config
    config = load_config()
    device = get_device()
    
    # Load face recognition model
    face_model = FaceRecognitionModel(device=device)
    face_model.load_employee_database(config['employee_db_path'])
    
    # Load metadata
    metadata_path = config['employee_db_path'].replace('.pkl', '_metadata.pkl')
    with open(metadata_path, 'rb') as f:
        db_metadata = pickle.load(f)
    
    # Load dataset
    images, targets, target_names = load_lfw_dataset(color=True, min_faces_per_person=20)
    
    # Initialize MTCNN
    mtcnn = MTCNN(image_size=160, margin=20, device=device, keep_all=False)
    
    # Find patches
    patches_dir = config['patches_dir']
    patch_files = [f for f in os.listdir(patches_dir) if f.endswith('.pt')]
    
    if not patch_files:
        print("ERROR: No patches found! Run 04_optimize_patch.py first.")
        return
    
    print(f"\nFound {len(patch_files)} patch(es)")
    
    evaluator = AttackEvaluator()
    
    # Test each patch
    for patch_file in patch_files:
        patch_path = os.path.join(patches_dir, patch_file)
        patch_tensor, patch_metadata = load_patch(patch_path)
        
        print("\n" + "="*70)
        print(f"TESTING PATCH: {patch_file}")
        print("="*70)
        print(f"Target: {patch_metadata['target_employee']}")
        print(f"Type: {patch_metadata['type']}")
        print(f"Size: {patch_metadata['size']} pixels")
        
        target_employee = patch_metadata['target_employee']
        
        # Get attacker images
        attacker_id = db_metadata['attacker_ids'][0]
        attacker_name = target_names[attacker_id]
        attacker_images = get_images_for_person(attacker_id, images, targets)
        
        # Use test images
        num_train = config['dataset']['attacker_train_images']
        num_test = config['dataset']['attacker_test_images']
        test_images = attacker_images[num_train:num_train + num_test]
        
        print(f"\nAttacker: {attacker_name}")
        print(f"Testing on {len(test_images)} images...")
        
        # Create circular mask
        if patch_metadata['type'] == 'circular':
            radius = patch_metadata['size'] // 2
            circular_mask = create_circular_mask(
                patch_metadata['size'],
                radius,
                device
            )
        else:
            circular_mask = torch.ones(1, patch_metadata['size'], patch_metadata['size']).to(device)
        
        # Test with patch
        predictions = []
        confidences = []
        
        for i, img in enumerate(test_images):
            # Detect face
            face = mtcnn(img)
            
            if face is not None:
                face = face.to(device)
                
                # Apply patch
                face_patched = apply_circular_patch(
                    face,
                    patch_tensor,
                    patch_metadata['position']['x'],
                    patch_metadata['position']['y'],
                    circular_mask
                )
                face_patched = torch.clamp(face_patched, -1, 1)
                
                # Get embedding
                face_patched_batch = face_patched.unsqueeze(0)
                with torch.no_grad():
                    patched_embedding = face_model.model(face_patched_batch)
                
                # Classify
                identified, confidence = face_model.classify_face(
                    patched_embedding,
                    threshold=config['classification_threshold']
                )
                
                predictions.append(identified)
                confidences.append(confidence)
                
                status = "✓ SUCCESS" if identified == target_employee else "✗ FAILED"
                print(f"  Test {i+1}: {identified:15s} (confidence: {confidence:.3f}) {status}")
            else:
                print(f"  Test {i+1}: [No face detected]")
        
        # Evaluate results
        results = evaluator.evaluate_targeted_attack(
            predictions,
            target=target_employee,
            confidences=confidences
        )
        
        print(f"\n{'='*70}")
        print("RESULTS")
        print(f"{'='*70}")
        print(f"Success Rate: {results['success_rate']:.1f}%")
        print(f"Successes: {results['successes']}/{results['total_tests']}")
        print(f"Avg Confidence: {results['avg_confidence']:.3f}")
        print(f"Avg Success Confidence: {results['avg_success_confidence']:.3f}")
        
        if results['success_rate'] > 50:
            print("\n✓ Attack is EFFECTIVE!")
        else:
            print("\n⚠ Attack needs improvement")
        
        # Update patch metadata with success rate
        patch_metadata['success_rate'] = results['success_rate']
        metadata_file = patch_path.replace('.pt', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(patch_metadata, f, indent=2)
        print(f"\n✓ Updated patch metadata with success rate")
    
    print("\n" + "="*70)
    print("✓ ATTACK TEST COMPLETE")
    print("="*70)
    print("\nNext step: Run 06_generate_report.py to create full report")


if __name__ == "__main__":
    main()