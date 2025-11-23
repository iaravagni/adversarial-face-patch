import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.utils.config import load_config, get_device
from src.data.dataset import load_lfw_dataset
from src.models.patch_detection_model import PatchDetector, PatchTrainingDataset

def main():
    # 1. Setup
    config = load_config()
    device = get_device() 

    # 2. Load Base Data (LFW)
    imgs_np, _, _ = load_lfw_dataset(min_faces_per_person=70)
    
    # Convert Numpy (N, H, W, 3) -> List of Tensors (3, 128, 128)
    print("Preprocessing images...")
    base_images = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor() # Converts to [0,1] and (C,H,W)
    ])

    for img in imgs_np:
        if img.dtype == 'float64' or img.dtype == 'float32':
            img = (img * 255).astype('uint8')
        
        tensor_img = transform(img)
        base_images.append(tensor_img)

    # 3. Create Patch Datasets
    train_data = PatchTrainingDataset(base_images, num_samples=2000) 
    test_data = PatchTrainingDataset(base_images, num_samples=500)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 4. Initialize Model
    model = PatchDetector().to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # 5. Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    best_acc = 0
    
    # Save path management
    save_dir = config.get('models_dir', 'models')
    if not os.path.isabs(save_dir):
         base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
         save_dir = os.path.join(base_path, save_dir)
         
    os.makedirs(save_dir, exist_ok=True)
    
    filename = config.get('patch_detector_filename', 'patch_detector.pth')
    save_path = os.path.join(save_dir, filename)

    print(f"\nStarting training... Models will be saved to: {save_path}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (pred == labels).sum().item()

        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                
                test_total += labels.size(0)
                test_correct += (pred == labels).sum().item()
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_acc = 100 * train_correct / train_total
        test_acc = 100 * test_correct / test_total

        print(f'Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model")

    # 6. Final Evaluation
    print(f"\nTraining Complete. Best Accuracy: {best_acc:.2f}%")
    print(f"Model saved to {save_path}")
    print("\nClassification Report on Test Set:")
    print(classification_report(all_labels, all_preds, target_names=['Clean', 'Patch']))

if __name__ == "__main__":
    main()