"""
Vision Transformer Model for AgriAI-ViT
Implements ViT-based architecture for crop disease detection
"""

import torch
import torch.nn as nn
from timm import create_model
from typing import Dict, Optional


class ViTClassifier(nn.Module):
      """
          Vision Transformer classifier for crop disease detection
              Uses pre-trained ViT from timm library
                  """

    def __init__(
              self,
              num_classes: int,
              model_name: str = 'vit_base_patch16_224',
              pretrained: bool = True,
              dropout: float = 0.1
    ):
              """
                      Initialize ViT classifier

                                      Args:
                                                  num_classes: Number of disease classes to predict
                                                              model_name: Name of the ViT model from timm
                                                                          pretrained: Whether to use ImageNet pretrained weights
                                                                                      dropout: Dropout rate for regularization
                                                                                              """
              super(ViTClassifier, self).__init__()

        # Load pre-trained Vision Transformer
              self.vit = create_model(
                  model_name,
                  pretrained=pretrained,
                  num_classes=0  # Remove classification head
              )

        # Get embedding dimension
              self.embed_dim = self.vit.embed_dim if hasattr(self.vit, 'embed_dim') else 768

        # Custom classification head
              self.classifier = nn.Sequential(
                  nn.Dropout(dropout),
                  nn.Linear(self.embed_dim, 512),
                  nn.GELU(),
                  nn.Dropout(dropout),
                  nn.Linear(512, num_classes)
              )

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
              """
                      Forward pass

                                      Args:
                                                  x: Input tensor of shape (batch_size, 3, 224, 224)

                                                                      Returns:
                                                                                  Logits of shape (batch_size, num_classes)
                                                                                          """
              # Extract features from ViT backbone
              features = self.vit(x)

        # Apply classification head
              logits = self.classifier(features)

        return logits

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
              """
                      Extract attention maps for visualization

                                      Args:
                                                  x: Input tensor

                                                                      Returns:
                                                                                  Attention maps from the last layer
                                                                                          """
              # This is a placeholder - implement based on specific ViT architecture
              return None


class ViTWithExplainability(ViTClassifier):
      """
          Enhanced ViT model with explainability features
              Supports Grad-CAM visualization
                  """

    def __init__(self, *args, **kwargs):
              super().__init__(*args, **kwargs)
              self.gradients = None
              self.activations = None

    def activations_hook(self, grad):
              """Hook to save gradients"""
              self.gradients = grad

    def forward_with_cam(self, x: torch.Tensor):
              """
                      Forward pass that saves activations for CAM

                                      Args:
                                                  x: Input tensor

                                                                      Returns:
                                                                                  Tuple of (logits, activations)
                                                                                          """
              # Get features from ViT
              features = self.vit.forward_features(x)

        # Register hook
        if features.requires_grad:
                      features.register_hook(self.activations_hook)

        self.activations = features

        # Pool and classify
        pooled = features.mean(dim=1)  # Global average pooling
        logits = self.classifier(pooled)

        return logits, features


def create_vit_model(
      num_classes: int,
      model_config: Optional[Dict] = None,
      device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> ViTClassifier:
      """
          Factory function to create ViT model

                  Args:
                          num_classes: Number of output classes
                                  model_config: Configuration dictionary
                                          device: Device to load model on

                                                      Returns:
                                                              Initialized ViT model
                                                                  """
      if model_config is None:
                model_config = {
                              'model_name': 'vit_base_patch16_224',
                              'pretrained': True,
                              'dropout': 0.1
                }

      model = ViTClassifier(num_classes=num_classes, **model_config)
      model = model.to(device)

    return model


def load_checkpoint(
      model: nn.Module,
      checkpoint_path: str,
      device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> nn.Module:
      """
          Load model from checkpoint

                  Args:
                          model: Model instance
                                  checkpoint_path: Path to checkpoint file
                                          device: Device to load on

                                                      Returns:
                                                              Model with loaded weights
                                                                  """
      checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
              model.load_state_dict(checkpoint['model_state_dict'])
else:
          model.load_state_dict(checkpoint)

    return model


if __name__ == "__main__":
      # Test model creation
      print("Testing ViT model creation...")

    num_classes = 10
    model = create_vit_model(num_classes=num_classes)

    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    if torch.cuda.is_available():
              dummy_input = dummy_input.cuda()

    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Model test successful!")
