"""
Build employee database from LFW dataset.
Creates face embeddings for all employees.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
import torch
from src.models.face_recognition import FaceRecognitionModel
from src.data.dataset import load_lfw_dataset, select_people_for_experiment, get_images_for_person, save_selected_people
from src.utils.config import load_config, get_device
import pickle


def main():
    """Build employee embedding database"""
    
    print("="*70)
    print("BUILDING EMPLOYEE DATABASE")
    print("="*70)
    
    # Load config
    config = load_config()
    device = get_device()
    
    # Load dataset
    images, targets, target_names = load_lfw_dataset(
        color=True,
        min_faces_per_person=30
    )
    
    print("\nSelecting employees and attackers...")
    employee_ids, attacker_ids = select_people_for_experiment(
        targets,
        target_names,
        num_employees=config["dataset"]["num_employees"],
        num_attackers=config["dataset"]["num_attackers"]
    )

    print("\nSaving selected images only...")
    save_selected_people(
        save_root=config["raw_data_dir"],
        images=images,
        targets=targets,
        target_names=target_names,
        employee_ids=employee_ids,
        attacker_ids=attacker_ids
    )

    print("\nSelected People (Employees + Attackers):")
    print("----------------------------------------")

    for pid in employee_ids:
        name = target_names[pid]
        count = np.sum(targets == pid)
        print(f"Employee – {name}: {count} images")

    for pid in attacker_ids:
        name = target_names[pid]
        count = np.sum(targets == pid)
        print(f"Attacker – {name}: {count} images")

    # Initialize face recognition model
    face_model = FaceRecognitionModel(device=device)
    
    # Process each employee
    print("\n" + "="*70)
    print("PROCESSING EMPLOYEES")
    print("="*70)
    
    for i, emp_id in enumerate(employee_ids):
        emp_name = target_names[emp_id]
        emp_label = f"Employee_{i+1}"
        
        print(f"\nProcessing {emp_label}: {emp_name}")
        
        # Get images for this employee
        person_images = get_images_for_person(emp_id, images, targets)
        
        # Use specified number of images
        num_to_use = min(
            config['dataset']['images_per_employee'],
            len(person_images)
        )
        images_to_use = person_images[:num_to_use]
        
        # Add to database
        face_model.add_employee(emp_label, images_to_use)
    
    # Save employee database
    os.makedirs(os.path.dirname(config['employee_db_path']), exist_ok=True)
    face_model.save_employee_database(config['employee_db_path'])
    
    # Also save metadata
    metadata = {
        'employee_ids': employee_ids,
        'attacker_ids': attacker_ids,
        'target_names': target_names,
        'num_employees': len(employee_ids),
        'num_attackers': len(attacker_ids)
    }
    
    metadata_path = config['employee_db_path'].replace('.pkl', '_metadata.pkl')
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n✓ Saved metadata to {metadata_path}")
    
    print("\n" + "="*70)
    print("✓ EMPLOYEE DATABASE BUILD COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()