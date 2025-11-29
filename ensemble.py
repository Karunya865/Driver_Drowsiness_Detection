# ensemble.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm
import os
import json
import glob
from PIL import Image
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
config = {
    'CLASSES': ['Closed_Eyes', 'Open_Eyes', 'yawn', 'no_yawn'],
    'NUM_CLASSES': 4,
    'IMG_SIZE': 224,
    'BATCH_SIZE': 16,
    'TEST_DATA_DIR': "dataset/dataset_eyes/test",
    'SAVE_DIR': "saved_models",
    'LOG_DIR': "ensemble_results"
}

# Create directories
os.makedirs(config['LOG_DIR'], exist_ok=True)

# Define the model classes
class EfficientNetDetector(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(EfficientNetDetector, self).__init__()
        self.model = models.efficientnet_b0(pretrained=pretrained)
        
        # Freeze all layers except the last MBConv block
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.features[-1].parameters():
            param.requires_grad = True
        
        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class ResNetDetector(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(ResNetDetector, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the last residual block (layer4)
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class VGGDetector(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(VGGDetector, self).__init__()
        self.model = models.vgg16(pretrained=pretrained)
        
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze Blocks 1-4 (layers 0 to 23)
        for param in self.model.features[:24].parameters():
            param.requires_grad = True
        
        # Keep Block 5 frozen (layers 24 to 30)
        for param in self.model.features[24:].parameters():
            param.requires_grad = False
        
        in_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Custom Dataset for test data
class TestDataset:
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load images and labels
        for class_idx, class_name in enumerate(config['CLASSES']):
            class_path = os.path.join(data_dir, class_name)
            if os.path.exists(class_path):
                images = glob.glob(os.path.join(class_path, '*.*'))
                self.image_paths.extend(images)
                self.labels.extend([class_idx] * len(images))
                print(f"Class {class_name}: {len(images)} images")
        
        print(f"Total test images: {len(self.image_paths)}")
    
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
            dummy_image = torch.randn(3, config['IMG_SIZE'], config['IMG_SIZE'])
            return dummy_image, label, "error"

class EnsembleModel:
    def __init__(self, model_paths, config):
        self.models = []
        self.config = config
        self.model_names = ['EfficientNet', 'ResNet', 'VGG']
        
        # Model mapping
        model_classes = {
            'efficient': EfficientNetDetector,
            'resnet': ResNetDetector,
            'vgg': VGGDetector
        }
        
        # Load all models
        for i, model_path in enumerate(model_paths):
            if os.path.exists(model_path):
                # Determine model type from filename
                model_type = None
                for key in model_classes.keys():
                    if key in model_path.lower():
                        model_type = key
                        break
                
                if model_type:
                    print(f"Loading {model_type} from {model_path}")
                    model = model_classes[model_type](num_classes=config['NUM_CLASSES'], pretrained=False)
                    
                    # Load model weights
                    checkpoint = torch.load(model_path, map_location=device)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model = model.to(device)
                    model.eval()
                    self.models.append(model)
                else:
                    print(f"Warning: Could not determine model type for {model_path}")
            else:
                print(f"Warning: Model path {model_path} does not exist")
        
        print(f"Loaded {len(self.models)} models for ensemble")
    
    def predict_proba(self, dataloader):
        """Get probability predictions from all models"""
        all_probs = [[] for _ in range(len(self.models))]
        all_labels = []
        all_paths = []
        
        with torch.no_grad():
            for images, labels, paths in tqdm(dataloader, desc='Ensemble Prediction'):
                images = images.to(device)
                
                # Get predictions from each model
                for i, model in enumerate(self.models):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    all_probs[i].append(probs.cpu().numpy())
                
                all_labels.extend(labels.numpy())
                all_paths.extend(paths)
        
        # Concatenate all predictions
        ensemble_probs = []
        for i in range(len(self.models)):
            all_probs[i] = np.concatenate(all_probs[i], axis=0)
            ensemble_probs.append(all_probs[i])
        
        return ensemble_probs, np.array(all_labels), all_paths
    
    def soft_voting(self, probs_list, weights=None):
        """Perform soft voting with optional model weights"""
        if weights is None:
            weights = [1.0] * len(probs_list)  # Equal weights
        
        # Weighted average of probabilities
        weighted_sum = np.zeros_like(probs_list[0])
        for i, probs in enumerate(probs_list):
            weighted_sum += probs * weights[i]
        
        # Normalize
        ensemble_probs = weighted_sum / sum(weights)
        ensemble_preds = np.argmax(ensemble_probs, axis=1)
        
        return ensemble_preds, ensemble_probs
    
    def evaluate_ensemble(self, dataloader, weights=None):
        """Evaluate ensemble performance"""
        # Get predictions from all models
        probs_list, true_labels, paths = self.predict_proba(dataloader)
        
        # Perform soft voting
        pred_labels, ensemble_probs = self.soft_voting(probs_list, weights)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        precision = precision_score(true_labels, pred_labels, average='weighted')
        recall = recall_score(true_labels, pred_labels, average='weighted')
        
        # Per-class metrics
        f1_per_class = f1_score(true_labels, pred_labels, average=None)
        precision_per_class = precision_score(true_labels, pred_labels, average=None)
        recall_per_class = recall_score(true_labels, pred_labels, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        report = classification_report(true_labels, pred_labels, 
                                     target_names=self.config['CLASSES'])
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'f1_per_class': f1_per_class,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'confusion_matrix': cm,
            'classification_report': report,
            'true_labels': true_labels,
            'pred_labels': pred_labels,
            'paths': paths,
            'ensemble_probs': ensemble_probs
        }

def plot_ensemble_metrics(results, config, save_dir='ensemble_results'):
    """Plot comprehensive metrics for ensemble model"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Confusion Matrix
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=config['CLASSES'],
                yticklabels=config['CLASSES'])
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Bar plot of overall metrics
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['accuracy'], results['precision'], 
              results['recall'], results['f1_score']]
    
    bars = ax2.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    ax2.set_title('Overall Metrics')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Per-class metrics
    classes = config['CLASSES']
    x = np.arange(len(classes))
    width = 0.25
    
    ax3.bar(x - width, results['precision_per_class'], width, label='Precision', alpha=0.8)
    ax3.bar(x, results['recall_per_class'], width, label='Recall', alpha=0.8)
    ax3.bar(x + width, results['f1_per_class'], width, label='F1-Score', alpha=0.8)
    
    ax3.set_xlabel('Classes')
    ax3.set_ylabel('Scores')
    ax3.set_title('Per-Class Metrics')
    ax3.set_xticks(x)
    ax3.set_xticklabels(classes, rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim(0, 1)
    
    # 4. Model comparison (if we have individual model results)
    ax4.text(0.5, 0.5, 'Ensemble Performance Summary\n\n' +
             f"Accuracy: {results['accuracy']:.4f}\n" +
             f"Precision: {results['precision']:.4f}\n" +
             f"Recall: {results['recall']:.4f}\n" +
             f"F1-Score: {results['f1_score']:.4f}",
             ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax4.set_title('Performance Summary')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ensemble_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional detailed plots
    plt.figure(figsize=(12, 8))
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, results['precision_per_class'], width, label='Precision', alpha=0.8)
    plt.bar(x, results['recall_per_class'], width, label='Recall', alpha=0.8)
    plt.bar(x + width, results['f1_per_class'], width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Scores')
    plt.title('Detailed Per-Class Metrics')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (p, r, f) in enumerate(zip(results['precision_per_class'], 
                                    results['recall_per_class'], 
                                    results['f1_per_class'])):
        plt.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
        plt.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
        plt.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detailed_per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_ensemble_results(results, config, save_dir='ensemble_results'):
    """Save ensemble results to files"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Value': [results['accuracy'], results['precision'], 
                 results['recall'], results['f1_score']]
    })
    metrics_df.to_csv(os.path.join(save_dir, 'overall_metrics.csv'), index=False)
    
    # Save per-class metrics
    per_class_df = pd.DataFrame({
        'Class': config['CLASSES'],
        'Precision': results['precision_per_class'],
        'Recall': results['recall_per_class'],
        'F1-Score': results['f1_per_class']
    })
    per_class_df.to_csv(os.path.join(save_dir, 'per_class_metrics.csv'), index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'image_path': results['paths'],
        'true_label': results['true_labels'],
        'predicted_label': results['pred_labels'],
        'true_class': [config['CLASSES'][label] for label in results['true_labels']],
        'predicted_class': [config['CLASSES'][pred] for pred in results['pred_labels']],
        'correct': [1 if true == pred else 0 for true, pred in zip(results['true_labels'], results['pred_labels'])]
    })
    predictions_df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)
    
    # Save classification report
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(results['classification_report'])
        f.write(f"\n\nOverall Accuracy: {results['accuracy']:.4f}")
        f.write(f"\nWeighted F1-Score: {results['f1_score']:.4f}")
        f.write(f"\nWeighted Precision: {results['precision']:.4f}")
        f.write(f"\nWeighted Recall: {results['recall']:.4f}")
    
    # Save confusion matrix
    cm_df = pd.DataFrame(results['confusion_matrix'], 
                        index=config['CLASSES'],
                        columns=config['CLASSES'])
    cm_df.to_csv(os.path.join(save_dir, 'confusion_matrix.csv'))
    
    print(f"Results saved to {save_dir}/")

def main():
    # Check if test directory exists
    if not os.path.exists(config['TEST_DATA_DIR']):
        print(f"Error: Test data directory '{config['TEST_DATA_DIR']}' does not exist.")
        print("Please update the config['TEST_DATA_DIR'] path in the script.")
        return
    
    # Define model paths - update these to match your actual model file names
    model_paths = [
        'saved_models/efficient_best_model.pth',
        'saved_models/resnet_best_model.pth', 
        'saved_models/vgg_best_model.pth'
    ]
    
    # Create test transforms
    test_transform = transforms.Compose([
        transforms.Resize((config['IMG_SIZE'], config['IMG_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset and loader
    test_dataset = TestDataset(config['TEST_DATA_DIR'], transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], 
                           shuffle=False, num_workers=2)
    
    # Create ensemble
    ensemble = EnsembleModel(model_paths, config)
    
    if len(ensemble.models) == 0:
        print("No models were loaded. Please check your model paths.")
        print("Available files in saved_models/:")
        if os.path.exists('saved_models'):
            print(os.listdir('saved_models'))
        return
    
    # Evaluate ensemble
    print("\nEvaluating ensemble model...")
    results = ensemble.evaluate_ensemble(test_loader)
    
    # Print results
    print("\n" + "="*50)
    print("ENSEMBLE MODEL RESULTS")
    print("="*50)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Plot and save results
    plot_ensemble_metrics(results, config)
    save_ensemble_results(results, config)
    
    print("\nEnsemble evaluation completed!")
    print(f"F1-Score: {results['f1_score']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"Precision: {results['precision']:.4f}")

if __name__ == "__main__":
    main()