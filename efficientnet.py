# driver_drowsiness_detection.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms, models
import torch.nn.functional as F

from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import glob
import json
import warnings
import random
from collections import Counter
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Configuration - using simple dictionary instead of class
config = {
    # Dataset paths - updated for separate train/test directories
    'TRAIN_DATA_DIR': "dataset/dataset_eyes/train",
    'TEST_DATA_DIR': "dataset/dataset_eyes/test",
    'CLASSES': ['Closed_Eyes', 'Open_Eyes', 'yawn', 'no_yawn'],  # Updated to match your folder names
    'NUM_CLASSES': 4,
    
    # Training parameters
    'BATCH_SIZE': 16,
    'LEARNING_RATE': 1e-4,
    'NUM_EPOCHS': 10,
    'IMG_SIZE': 224,
    'PATIENCE': 5,  # For early stopping
    
    # Model parameters
    'PRETRAINED': True,
    'DROPOUT': 0.7,  # Increased dropout
    
    # Regularization
    'WEIGHT_DECAY': 1e-5,
    
    # Augmentation parameters
    'MIXUP_ALPHA': 0.2,
    'CUTMIX_ALPHA': 1.0,
    
    # Paths
    'SAVE_DIR': "saved_models",
    'LOG_DIR': "logs"
}

# Create directories
os.makedirs(config['SAVE_DIR'], exist_ok=True)
os.makedirs(config['LOG_DIR'], exist_ok=True)

# Custom Dataset
class DrowsinessDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=False):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.image_paths = []
        self.labels = []
        self.subject_ids = []  # For person-aware splitting
        
        # Load images and labels
        for class_idx, class_name in enumerate(config['CLASSES']):
            class_path = os.path.join(data_dir, class_name)
            if os.path.exists(class_path):
                images = glob.glob(os.path.join(class_path, '*.*'))
                self.image_paths.extend(images)
                self.labels.extend([class_idx] * len(images))
                
                # Extract subject IDs from filenames (assuming format like 'subject1_image1.jpg')
                for img_path in images:
                    filename = os.path.basename(img_path)
                    subject_id = filename.split('_')[0]  # Adjust this based on your filename format
                    self.subject_ids.append(subject_id)
                
                print(f"Class {class_name}: {len(images)} images")
        
        print(f"Total images loaded: {len(self.image_paths)}")
        print(f"Class distribution: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label, os.path.basename(img_path)
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image if there's an error
            dummy_image = torch.randn(3, config['IMG_SIZE'], config['IMG_SIZE'])
            return dummy_image, label, "error"

# Advanced data transformations with stronger augmentations
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((config['IMG_SIZE'] + 32, config['IMG_SIZE'] + 32)),
        transforms.RandomCrop(config['IMG_SIZE']),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((config['IMG_SIZE'], config['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

# Mixup augmentation
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# CutMix augmentation
def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    # Generate random bounding box
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y, y[index], lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

# EfficientNet Model with reduced classifier
class DrowsinessDetector(nn.Module):
    def __init__(self, num_classes=config['NUM_CLASSES'], pretrained=True):
        super(DrowsinessDetector, self).__init__()
        
        # Use EfficientNet which is widely available
        self.model = models.efficientnet_b0(pretrained=pretrained)
        
        # Freeze all layers except the last MBConv block
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last MBConv block
        for param in self.model.features[-1].parameters():
            param.requires_grad = True
        
        # Replace the classifier with a smaller one
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(config['DROPOUT']),
            nn.Linear(in_features, num_classes)  # Reduced from 256 to direct classification
        )
    
    def forward(self, x):
        return self.model(x)

# Training function with advanced augmentations
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    best_acc = 0.0
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, labels, _) in enumerate(train_pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            # Apply Mixup or CutMix randomly
            if random.random() < 0.5:
                # Mixup
                images, targets_a, targets_b, lam = mixup_data(images, labels, config['MIXUP_ALPHA'])
                optimizer.zero_grad()
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                # CutMix
                images, targets_a, targets_b, lam = cutmix_data(images, labels, config['CUTMIX_ALPHA'])
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (1. - lam)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': 100.*correct/total
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch_idx, (images, labels, _) in enumerate(val_pbar):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': val_loss/(batch_idx+1),
                    'acc': 100.*correct/total
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        if scheduler:
            scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= config['PATIENCE']:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, os.path.join(config['SAVE_DIR'], 'best_model.pth'))
            print(f'New best model saved with accuracy: {val_acc:.2f}%')
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }
    
    return history

# Calculate class weights for imbalanced data
def get_class_weights(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    weights = []
    for class_idx in range(num_classes):
        if class_idx in class_counts:
            weight = total_samples / (num_classes * class_counts[class_idx])
            weights.append(weight)
        else:
            weights.append(1.0)  # If class is missing, use default weight
    
    return torch.FloatTensor(weights).to(device)

# Person-aware train/val split (only for training data)
def person_aware_split(dataset, val_size=0.2, random_state=42):
    # Group images by subject
    subject_groups = {}
    for i, subject_id in enumerate(dataset.subject_ids):
        if subject_id not in subject_groups:
            subject_groups[subject_id] = []
        subject_groups[subject_id].append(i)
    
    # Get unique subjects
    subjects = list(subject_groups.keys())
    
    # Split subjects into train and val
    train_subjects, val_subjects = train_test_split(
        subjects, test_size=val_size, random_state=random_state
    )
    
    # Get indices for each split
    train_indices = []
    for subject in train_subjects:
        train_indices.extend(subject_groups[subject])
    
    val_indices = []
    for subject in val_subjects:
        val_indices.extend(subject_groups[subject])
    
    return train_indices, val_indices

# Main training function
def main():
    print("Starting Driver Drowsiness Detection Training...")
    print(f"Number of classes: {config['NUM_CLASSES']}")
    print(f"Classes: {config['CLASSES']}")
    
    # Check if data directories exist
    if not os.path.exists(config['TRAIN_DATA_DIR']):
        print(f"Error: Training data directory '{config['TRAIN_DATA_DIR']}' does not exist.")
        return
    
    if not os.path.exists(config['TEST_DATA_DIR']):
        print(f"Warning: Test data directory '{config['TEST_DATA_DIR']}' does not exist.")
        print("Will use validation set for final evaluation.")
        use_test_set = False
    else:
        use_test_set = True
    
    # Get data transforms
    train_transform, val_transform = get_transforms()
    
    # Create training dataset
    train_dataset = DrowsinessDataset(config['TRAIN_DATA_DIR'], transform=train_transform, is_train=True)
    
    if len(train_dataset) == 0:
        print("No images found in the training directory.")
        return
    
    # Person-aware split for training data
    train_indices, val_indices = person_aware_split(train_dataset, val_size=0.2)
    
    # Create training and validation subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(train_dataset, val_indices)
    
    # Apply validation transform to validation set
    val_subset.dataset.transform = val_transform
    
    # Calculate class weights for imbalanced data
    train_labels = [train_dataset.labels[i] for i in train_indices]
    class_weights = get_class_weights(train_labels)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create weighted sampler for training
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders for training and validation
    train_loader = DataLoader(train_subset, batch_size=config['BATCH_SIZE'], 
                            sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=config['BATCH_SIZE'], 
                          shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Class weights: {class_weights.cpu().numpy()}")
    
    # Create test dataset if available
    if use_test_set:
        test_dataset = DrowsinessDataset(config['TEST_DATA_DIR'], transform=val_transform, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], 
                               shuffle=False, num_workers=2, pin_memory=True)
        print(f"Test samples: {len(test_dataset)}")
    else:
        test_loader = None
    
    # Use EfficientNet model
    model = DrowsinessDetector(num_classes=config['NUM_CLASSES'], 
                             pretrained=config['PRETRAINED'])
    model = model.to(device)
    
    # Optimizer with weight decay
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['LEARNING_RATE'], 
        weight_decay=config['WEIGHT_DECAY']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train the model
    history = train_model(model, train_loader, val_loader, criterion, 
                         optimizer, scheduler, config['NUM_EPOCHS'])
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model for evaluation
    best_model_path = os.path.join(config['SAVE_DIR'], 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model with validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate on test set if available, otherwise use validation set
    if test_loader:
        print("\nEvaluating on test set...")
        accuracy, cm, report, all_preds, all_labels, all_paths = evaluate_model(model, test_loader)
        results_name = 'test'
    else:
        print("\nEvaluating on validation set (test set not available)...")
        accuracy, cm, report, all_preds, all_labels, all_paths = evaluate_model(model, val_loader)
        results_name = 'validation'
    
    # Save results
    results_df = pd.DataFrame({
        'image_path': all_paths,
        'true_label': all_labels,
        'predicted_label': all_preds,
        'true_class': [config['CLASSES'][label] for label in all_labels],
        'predicted_class': [config['CLASSES'][pred] for pred in all_preds],
        'correct': [1 if true == pred else 0 for true, pred in zip(all_labels, all_preds)]
    })
    
    results_df.to_csv(os.path.join(config['LOG_DIR'], f'{results_name}_results.csv'), index=False)
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'classes': config['CLASSES'],
        'accuracy': accuracy,
        'class_weights': class_weights.cpu().numpy()
    }, os.path.join(config['SAVE_DIR'], 'final_model.pth'))
    
    # Save config to JSON
    with open(os.path.join(config['SAVE_DIR'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"\nTraining completed! Final {results_name} accuracy: {accuracy:.4f}")

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, 
                                 target_names=config['CLASSES'])
    
    print(f'Accuracy: {accuracy:.4f}')
    print('\nConfusion Matrix:')
    print(cm)
    print('\nClassification Report:')
    print(report)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config['CLASSES'],
                yticklabels=config['CLASSES'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(config['LOG_DIR'], 'confusion_matrix.png'))
    plt.close()  # Close to avoid display issues
    
    return accuracy, cm, report, all_preds, all_labels, all_paths

# Plot training history
def plot_training_history(history):
    epochs = range(1, len(history['train_losses']) + 1) 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_accs'], label='Train Accuracy')
    ax2.plot(history['val_accs'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['LOG_DIR'], 'training_history.png'))
    plt.close()  # Close to avoid display issues

if __name__ == "__main__":
    main()