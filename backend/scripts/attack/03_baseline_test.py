"""
Test baseline face recognition performance (no attack).
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import pickle
from src.models.face_recognition import FaceRecognitionModel
from src.data.dataset import load_lfw_dataset, get_images_for_person
from src.utils.config import load_config, get_device
from src.attack.evaluator import AttackEvaluator


def main():
    """Test baseline performance"""
    
    print("="*70)
    print("BASELINE TEST (No Attack)")
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
        metadata = pickle.load(f)
    
    # Load dataset
    images, targets, target_names = load_lfw_dataset(color=True, min_faces_per_person=20)
    
    # Test on attackers
    print("\n" + "="*70)
    print("TESTING ATTACKERS (Should be recognized as 'Unknown')")
    print("="*70)
    
    evaluator = AttackEvaluator()
    
    for attacker_id in metadata['attacker_ids']:
        attacker_name = target_names[attacker_id]
        attacker_images = get_images_for_person(attacker_id, images, targets)
        
        # Use test images
        num_train = config['dataset']['attacker_train_images']
        num_test = config['dataset']['attacker_test_images']
        test_images = attacker_images[num_train:num_train + num_test]
        
        print(f"\nAttacker: {attacker_name}")
        print(f"Testing on {len(test_images)} images...")
        
        predictions = []
        confidences = []
        
        for i, img in enumerate(test_images):
            embedding = face_model.get_face_embedding(img)
            
            if embedding is not None:
                identified, confidence = face_model.classify_face(
                    embedding,
                    threshold=config['classification_threshold']
                )
                predictions.append(identified)
                confidences.append(confidence)
                
                print(f"  Test {i+1}: {identified:15s} (confidence: {confidence:.3f})")
            else:
                print(f"  Test {i+1}: [No face detected]")
        
        # Evaluate
        results = evaluator.evaluate_baseline(
            predictions,
            expected_identity="Unknown",
            confidences=confidences
        )
        
        print(f"\nResults for {attacker_name}:")
        print(f"  Accuracy: {results['accuracy']:.1f}%")
        print(f"  Correct (Unknown): {results['correct']}/{results['total_tests']}")
        print(f"  Avg Confidence: {results['avg_confidence']:.3f}")
    
    # Test on employees
    print("\n" + "="*70)
    print("TESTING EMPLOYEES (Should be recognized correctly)")
    print("="*70)
    
    employee_ids = metadata['employee_ids']
    
    for i, emp_id in enumerate(employee_ids[:3]):  # Test first 3 employees
        emp_name = target_names[emp_id]
        emp_label = f"Employee_{i+1}"
        emp_images = get_images_for_person(emp_id, images, targets)
        
        # Use images not in training
        num_train = config['dataset']['images_per_employee']
        test_images = emp_images[num_train:num_train + 10]
        
        print(f"\n{emp_label}: {emp_name}")
        print(f"Testing on {len(test_images)} images...")
        
        predictions = []
        confidences = []
        
        for j, img in enumerate(test_images):
            embedding = face_model.get_face_embedding(img)
            
            if embedding is not None:
                identified, confidence = face_model.classify_face(
                    embedding,
                    threshold=config['classification_threshold']
                )
                predictions.append(identified)
                confidences.append(confidence)
                
                status = "✓" if identified == emp_label else "✗"
                print(f"  Test {j+1}: {identified:15s} (confidence: {confidence:.3f}) {status}")
            else:
                print(f"  Test {j+1}: [No face detected]")
        
        # Evaluate
        results = evaluator.evaluate_baseline(
            predictions,
            expected_identity=emp_label,
            confidences=confidences
        )
        
        print(f"\nResults for {emp_label}:")
        print(f"  Accuracy: {results['accuracy']:.1f}%")
        print(f"  Correct: {results['correct']}/{results['total_tests']}")
        print(f"  Avg Confidence: {results['avg_confidence']:.3f}")
    
    print("\n" + "="*70)
    print("✓ BASELINE TEST COMPLETE")
    print("="*70)
    print("\nNext step: Run 04_optimize_patch.py to create adversarial patches")


if __name__ == "__main__":
    main()