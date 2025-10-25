"""
AgriAI-ViT Training Script
Vision Transformer training with Comet ML tracking
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import comet_ml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from model import ViTClassifier
from dataset import CropDiseaseDataset, get_transforms
from utils import EarlyStopping, save_checkpoint, load_checkpoint


def train_epoch(model, dataloader, criterion, optimizer, device, experiment):
      """Train for one epoch"""
      model.train()
      running_loss = 0.0
      all_preds = []
      all_labels = []

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
              images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, all_preds, all_labels


def validate_epoch(model, dataloader, criterion, device, experiment):
      """Validate for one epoch"""
      model.eval()
      running_loss = 0.0
      all_preds = []
      all_labels = []

    with torch.no_grad():
              pbar = tqdm(dataloader, desc='Validation')
              for images, labels in pbar:
                            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, class_names, experiment, epoch):
      """Plot and log confusion matrix"""
      cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    experiment.log_figure(figure_name=f'confusion_matrix_epoch_{epoch}', figure=plt)
    plt.close()


def main():
      # Initialize Comet experiment
      experiment = comet_ml.Experiment(
                api_key=os.getenv('COMET_API_KEY'),
                project_name='AgriAI-ViT',
                workspace=os.getenv('COMET_WORKSPACE', 'agriai')
      )

    # Log configuration
      config = Config()
      experiment.log_parameters(config.__dict__)

    # Set device
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      print(f'Using device: {device}')
      experiment.log_parameter('device', str(device))

    # Create datasets
      train_transform = get_transforms(mode='train')
      val_transform = get_transforms(mode='val')

    train_dataset = CropDiseaseDataset(
              root_dir=config.data_dir,
              split='train',
              transform=train_transform
    )

    val_dataset = CropDiseaseDataset(
              root_dir=config.data_dir,
              split='val',
              transform=val_transform
    )

    # Create dataloaders
    train_loader = DataLoader(
              train_dataset,
              batch_size=config.batch_size,
              shuffle=True,
              num_workers=config.num_workers,
              pin_memory=True
    )

    val_loader = DataLoader(
              val_dataset,
              batch_size=config.batch_size,
              shuffle=False,
              num_workers=config.num_workers,
              pin_memory=True
    )

    # Create model
    num_classes = """
AgriAI-ViT Training Script
Vision Transformer training with Comet ML tracking
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import comet_ml
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from model import ViTClassifier
from dataset import CropDiseaseDataset, get_transforms
from utils import EarlyStopping, save_checkpoint, load_checkpoint


def train_epoch(model, dataloader, criterion, optimizer, device, experiment):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def validate_epoch(model, dataloader, criterion, device, experiment):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, class_names, experiment, epoch):
    """Plot and log confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    experiment.log_figure(figure_name=f'confusion_matrix_epoch_{epoch}', figure=plt)
    plt.close()


def main():
    # Initialize Comet experiment
    experiment = comet_ml.Experiment(
        api_key=os.getenv('COMET_API_KEY'),
        project_name='AgriAI-ViT',
        workspace=os.getenv('COMET_WORKSPACE', 'agriai')
    )
    
    # Log configuration
    config = Config()
    experiment.log_parameters(config.__dict__)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    experiment.log_parameter('device', str(device))
    
    # Create datasets
    train_transform = get_transforms(mode='train')
    val_transform = get_transforms(mode='val')
    
    train_dataset = CropDiseaseDataset(
        root_dir=config.data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = CropDiseaseDataset(
        root_dir=config.data_dir,
        split='val',
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # Create model
    num_classes = len(train_dataset.classes)
    model = ViTClassifier(num_classes=num_classes, model_name=config.model_name)
    model = model.to(device)
    
    # Log model summary
    experiment.set_model_graph(str(model))
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(config.num_epochs):
        print(f'\nEpoch {epoch+1}/{config.num_epochs}')
        print('-' * 60)
        
        # Train
        train_loss, train_acc, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device, experiment
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device, experiment
        )
        
        # Calculate metrics
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='weighted', zero_division=0
        )
        
        # Log metrics
        experiment.log_metrics({
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'learning_rate': optimizer.param_groups[0]['lr']
        }, epoch=epoch+1)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')
        
        # Plot confusion matrix every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_confusion_matrix(
                val_labels, val_preds, 
                train_dataset.classes, 
                experiment, epoch+1
            )
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_names': train_dataset.classes
            }, filename=os.path.join(config.checkpoint_dir, 'best_model.pt'))
            print(f'Saved best model with val_acc: {val_acc:.4f}')
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print('Early stopping triggered')
            break
    
    # Final evaluation
    print('\nTraining completed!')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    experiment.log_metric('best_val_accuracy', best_val_acc)
    
    experiment.end()


if __name__ == '__main__':
    main()
(train_dataset.classes)
    model = ViTClassifier(num_classes=num_classes, model_name=config.model_name)
    model = model.to(device)

    # Log model summary
    experiment.set_model_graph(str(model))

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
              model.parameters(),
              lr=config.learning_rate,
              weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
              optimizer,
              mode='min',
              factor=0.1,
              patience=3,
              verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, verbose=True)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(config.num_epochs):
              print(f'\nEpoch {epoch+1}/{config.num_epochs}')
              print('-' * 60)

        # Train
              train_loss, train_acc, train_preds, train_labels = train_epoch(
                  model, train_loader, criterion, optimizer, device, experiment
              )

        # Validate
              val_loss, val_acc, val_preds, val_labels = validate_epoch(
                  model, val_loader, criterion, device, experiment
              )

        # Calculate metrics
              val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                  val_labels, val_preds, average='weighted', zero_division=0
              )

        # Log metrics
              experiment.log_metrics({
                  'train_loss': train_loss,
                  'train_accuracy': train_acc,
                  'val_loss': val_loss,
                  'val_accuracy': val_acc,
                  'val_precision': val_precision,
                  'val_recall': val_recall,
                  'val_f1': val_f1,
                  'learning_rate': optimizer.param_groups[0]['lr']
              }, epoch=epoch+1)

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}')

        # Plot confusion matrix every 5 epochs
        if (epoch + 1) % 5 == 0:
                      plot_confusion_matrix(
                                        val_labels, val_preds, 
                                        train_dataset.classes, 
                                        experiment, epoch+1
                      )

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
                      best_val_acc = val_acc
                      save_checkpoint({
                          'epoch': epoch + 1,
                          'state_dict': model.state_dict(),
                          'optimizer': optimizer.state_dict(),
                          'val_acc': val_acc,
                          'val_loss': val_loss,
                         
