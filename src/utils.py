"""
AgriAI-ViT Utility Functions
Provides helper functions for visualization, metrics, and Grad-CAM
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_fscore_support, roc_curve, auc
)
from typing import List, Tuple, Dict
import cv2
from pathlib import Path


class GradCAM:
      """
          Gradient-weighted Class Activation Mapping (Grad-CAM) for Vision Transformers
              Provides explainability by highlighting important regions in the input image
                  """

    def __init__(self, model, target_layer):
              """
                      Initialize Grad-CAM

                                      Args:
                                                  model: The Vision Transformer model
                                                              target_layer: Target layer for gradient computation
                                                                      """
              self.model = model
              self.target_layer = target_layer
              self.gradients = None
              self.activations = None

        # Register hooks
              self.target_layer.register_forward_hook(self.save_activation)
              self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
              """Save forward pass activations"""
              self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
              """Save backward pass gradients"""
              self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
              """
                      Generate Class Activation Map

                                      Args:
                                                  input_image: Input tensor (1, C, H, W)
                                                              target_class: Target class index (if None, use predicted class)

                                                                                  Returns:
                                                                                              cam: Class activation map
                                                                                                          prediction: Model prediction
                                                                                                                  """
              # Forward pass
              self.model.eval()
              output = self.model(input_image)

        if target_class is None:
                      target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
                      cam += w * activations[i]

        # Apply ReLU
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.cpu().numpy(), target_class

    def overlay_cam_on_image(self, image, cam, alpha=0.5):
              """
                      Overlay CAM heatmap on original image

                                      Args:
                                                  image: Original image (H, W, C) in range [0, 255]
                                                              cam: Class activation map (H', W')
                                                                          alpha: Transparency factor

                                                                                              Returns:
                                                                                                          overlaid_image: Image with CAM overlay
                                                                                                                  """
              # Resize CAM to match image size
              cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))

        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay heatmap on image
        overlaid = heatmap * alpha + image * (1 - alpha)
        overlaid = np.uint8(overlaid)

        return overlaid, heatmap


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                                                   class_names: List[str], save_path: str = None) -> plt.Figure:
                                                         """
                                                             Plot confusion matrix

                                                                     Args:
                                                                             y_true: True labels
                                                                                     y_pred: Predicted labels
                                                                                             class_names: List of class names
                                                                                                     save_path: Path to save the figure
                                                                                                             
                                                                                                                 Returns:
                                                                                                                         fig: Matplotlib figure
                                                                                                                             """
                                                         cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=class_names, yticklabels=class_names,
                                ax=ax, cbar_kws={'label': 'Count'})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
              plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_training_curves(train_losses: List[float], val_losses: List[float],
                                                  train_accs: List[float], val_accs: List[float],
                                                  save_path: str = None) -> plt.Figure:
                                                        """
                                                            Plot training and validation curves

                                                                    Args:
                                                                            train_losses: Training losses per epoch
                                                                                    val_losses: Validation losses per epoch
                                                                                            train_accs: Training accuracies per epoch
                                                                                                    val_accs: Validation accuracies per epoch
                                                                                                            save_path: Path to save the figure
                                                                                                                    
                                                                                                                        Returns:
                                                                                                                                fig: Matplotlib figure
                                                                                                                                    """
                                                        epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot losses
    ax1.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-o', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-s', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
              plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_class_distribution(dataset, class_names: List[str], 
                                                       save_path: str = None) -> plt.Figure:
                                                             """
                                                                 Plot class distribution in dataset

                                                                         Args:
                                                                                 dataset: PyTorch dataset with targets attribute
                                                                                         class_names: List of class names
                                                                                                 save_path: Path to save the figure
                                                                                                         
                                                                                                             Returns:
                                                                                                                     fig: Matplotlib figure
                                                                                                                         """
                                                             if hasattr(dataset, 'targets'):
                                                                       labels = dataset.targets
else:
        labels = [dataset[i][1] for i in range(len(dataset))]

    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(class_names, counts, color='steelblue', alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
              height = bar.get_height()
              ax.text(bar.get_x() + bar.get_width()/2., height,
                      f'{int(height)}',
                      ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    if save_path:
              plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                           class_names: List[str]) -> Dict:
                                                 """
                                                     Calculate comprehensive evaluation metrics

                                                             Args:
                                                                     y_true: True labels
                                                                             y_pred: Predicted labels
                                                                                     class_names: List of class names

                                                                                                 Returns:
                                                                                                         metrics_dict: Dictionary containing all metrics
                                                                                                             """
                                                 # Overall metrics
                                                 precision, recall, f1, support = precision_recall_fscore_support(
                                                     y_true, y_pred, average='weighted'
                                                 )

    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(y_true, y_pred, average=None)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                                                      output_dict=True)

    metrics_dict = {
              'overall': {
                            'precision': float(precision),
                            'recall': float(recall),
                            'f1_score': float(f1),
                            'accuracy': float((y_true == y_pred).mean())
              },
              'per_class': {
                            class_names[i]: {
                                              'precision': float(per_class_precision[i]),
                                              'recall': float(per_class_recall[i]),
                                              'f1_score': float(per_class_f1[i]),
                                              'support': int(per_class_support[i])
                            }
                            for i in range(len(class_names))
              },
              'classification_report': report
    }

    return metrics_dict


def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, 
                                       class_names: List[str], save_path: str = None) -> plt.Figure:
                                             """
                                                 Plot ROC curves for multiclass classification

                                                         Args:
                                                                 y_true: True labels (one-hot encoded)
                                                                         y_probs: Predicted probabilities
                                                                                 class_names: List of class names
                                                                                         save_path: Path to save the figure

                                                                                                     Returns:
                                                                                                             fig: Matplotlib figure
                                                                                                                 """
                                             n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, n_classes))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
              fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
              roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=color, linewidth=2,
                                label=f'{class_name} (AUC = {roc_auc:.2f})')

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Multiclass', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
              plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def visualize_predictions(images: torch.Tensor, true_labels: List[int], 
                                                   pred_labels: List[int], class_names: List[str],
                                                   num_images: int = 16, save_path: str = None) -> plt.Figure:
                                                         """
                                                             Visualize sample predictions

                                                                     Args:
                                                                             images: Batch of images (N, C, H, W)
                                                                                     true_labels: True labels
                                                                                             pred_labels: Predicted labels
                                                                                                     class_names: List of class names
                                                                                                             num_images: Number of images to display
                                                                                                                     save_path: Path to save the figure
                                                                                                                             
                                                                                                                                 Returns:
                                                                                                                                         fig: Matplotlib figure
                                                                                                                                             """
                                                         num_images = min(num_images, len(images))
                                                         rows = int(np.sqrt(num_images))
                                                         cols = int(np.ceil(num_images / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten() if num_images > 1 else [axes]

    for idx in range(num_images):
              # Denormalize image
              img = images[idx].cpu().numpy().transpose(1, 2, 0)
              mean = np.array([0.485, 0.456, 0.406])
              std = np.array([0.229, 0.224, 0.225])
              img = std * img + mean
              img = np.clip(img, 0, 1)

        axes[idx].imshow(img)

        true_class = class_names[true_labels[idx]]
        pred_class = class_names[pred_labels[idx]]

        color = 'green' if true_labels[idx] == pred_labels[idx] else 'red'
        axes[idx].set_title(f'True: {true_class}\nPred: {pred_class}', 
                                                       color=color, fontsize=10)
        axes[idx].axis('off')

    # Hide unused subplots
    for idx in range(num_images, len(axes)):
              axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
              plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


class EarlyStopping:
      """
          Early stopping to stop training when validation loss doesn't improve
              """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, 
                                  mode: str = 'min', verbose: bool = True):
                                            """
                                                    Initialize Early Stopping

                                                                    Args:
                                                                                patience: Number of epochs to wait before stopping
                                                                                            min_delta: Minimum change to qualify as improvement
                                                                                                        mode: 'min' for loss, 'max' for accuracy
                                                                                                                    verbose: Print messages
                                                                                                                            """
                                            self.patience = patience
                                            self.min_delta = min_delta
                                            self.mode = mode
                                            self.verbose = verbose
                                            self.counter = 0
                                            self.best_score = None
                                            self.early_stop = False
                                            self.best_epoch = 0

    def __call__(self, metric: float, epoch: int) -> bool:
              """
                      Check if training should stop

                                      Args:
                                                  metric: Current metric value
                                                              epoch: Current epoch number

                                                                                  Returns:
                                                                                              True if training should stop
                                                                                                      """
              score = -metric if self.mode == 'min' else metric

        if self.best_score is None:
                      self.best_score = score
                      self.best_epoch = epoch
elif score < self.best_score + self.min_delta:
              self.counter += 1
              if self.verbose:
                                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
                            if self.counter >= self.patience:
                                              self.early_stop = True
                                              if self.verbose:
                                                                    print(f'Early stopping triggered at epoch {epoch}')
                                                                    print(f'Best score: {self.best_score:.4f} at epoch {self.best_epoch}')
                            else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop


def save_checkpoint(model, optimizer, scheduler, epoch: int, 
                                       metrics: Dict, save_path: str):
                                             """
                                                 Save model checkpoint

                                                         Args:
                                                                 model: PyTorch model
                                                                         optimizer: Optimizer
                                                                                 scheduler: Learning rate scheduler
                                                                                         epoch: Current epoch
                                                                                                 metrics: Dictionary of metrics
                                                                                                         save_path: Path to save checkpoint
                                                                                                             """
                                             checkpoint = {
                                                 'epoch': epoch,
                                                 'model_state_dict': model.state_dict(),
                                                 'optimizer_state_dict': optimizer.state_dict(),
                                                 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                                                 'metrics': metrics
                                             }

    torch.save(checkpoint, save_path)
    print(f'Checkpoint saved to {save_path}')


def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str, 
                                       device: torch.device):
                                             """
                                                 Load model checkpoint

                                                         Args:
                                                                 model: PyTorch model
                                                                         optimizer: Optimizer
                                                                                 scheduler: Learning rate scheduler
                                                                                         checkpoint_path: Path to checkpoint
                                                                                                 device: Device to load model to
                                                                                                         
                                                                                                             Returns:
                                                                                                                     epoch: Epoch number
                                                                                                                             metrics: Dictionary of metrics
                                                                                                                                 """
                                             checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint['scheduler_state_dict']:
              scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']

    print(f'Checkpoint loaded from {checkpoint_path}')
    print(f'Resuming from epoch {epoch}')

    return epoch, metrics


if __name__ == '__main__':
      print("AgriAI-ViT Utils Module")
    print("Provides utility functions for visualization, metrics, and Grad-CAM")
