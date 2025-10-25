"""
AgriAI-ViT Configuration Module
Central configuration file for all hyperparameters, paths, and settings
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class PathConfig:
      """Configuration for file paths"""
      # Root directories
      project_root: Path = Path(__file__).parent.parent
      data_dir: Path = project_root / "data"
      models_dir: Path = project_root / "models"
      logs_dir: Path = project_root / "logs"

    # Data subdirectories
      raw_data_dir: Path = data_dir / "raw"
      processed_data_dir: Path = data_dir / "processed"

    # Kaggle dataset paths
      rice_dataset_path: Path = raw_data_dir / "rice-leaf-diseases"
      wheat_dataset_path: Path = raw_data_dir / "plantvillage-wheat"

    # Model checkpoints
      checkpoint_dir: Path = models_dir / "checkpoints"
      best_model_path: Path = models_dir / "best_model.pt"

    def create_directories(self):
              """Create all necessary directories"""
              dirs = [
                  self.data_dir,
                  self.models_dir,
                  self.logs_dir,
                  self.raw_data_dir,
                  self.processed_data_dir,
                  self.checkpoint_dir
              ]
              for dir_path in dirs:
                            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
      """Configuration for Vision Transformer model"""
      # Model architecture
      model_name: str = "vit_base_patch16_224"
      pretrained: bool = True
      num_classes: int = 10  # Will be updated based on dataset
    img_size: int = 224
    patch_size: int = 16

    # Model parameters
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Fine-tuning
    freeze_backbone: bool = False
    unfreeze_layers: int = 4  # Number of layers to unfreeze from the end


@dataclass
class TrainingConfig:
      """Configuration for training process"""
      # Training hyperparameters
      batch_size: int = 32
      num_epochs: int = 30
      learning_rate: float = 3e-5
      weight_decay: float = 0.01

    # Optimizer settings
      optimizer: str = "adamw"
      betas: tuple = (0.9, 0.999)
      eps: float = 1e-8

    # Learning rate scheduler
      scheduler: str = "reduce_on_plateau"
      scheduler_factor: float = 0.5
      scheduler_patience: int = 3
      scheduler_min_lr: float = 1e-7

    # Early stopping
      early_stopping: bool = True
      early_stopping_patience: int = 5
      early_stopping_delta: float = 0.001

    # Mixed precision training
      use_amp: bool = True

    # Gradient clipping
      grad_clip_norm: float = 1.0

    # Data loading
      num_workers: int = 4
      pin_memory: bool = True

    # Validation
      val_frequency: int = 1  # Validate every N epochs
    save_frequency: int = 5  # Save checkpoint every N epochs


@dataclass
class DataConfig:
      """Configuration for data processing"""
      # Data splits
      train_split: float = 0.7
      val_split: float = 0.2
      test_split: float = 0.1

    # Image preprocessing
      img_size: int = 224
      normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
      normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    # Data augmentation (for training)
      random_crop: bool = True
      random_horizontal_flip: bool = True
      random_rotation: int = 15
      color_jitter: bool = True
      color_jitter_params: Dict[str, float] = field(default_factory=lambda: {
          'brightness': 0.2,
          'contrast': 0.2,
          'saturation': 0.2,
          'hue': 0.1
      })

    # Class balancing
      use_class_weights: bool = True
      oversample_minority: bool = False


@dataclass
class CometConfig:
      """Configuration for Comet ML experiment tracking"""
      # Comet settings
      project_name: str = "agriai-vit"
      workspace: Optional[str] = None
      api_key: Optional[str] = os.getenv("COMET_API_KEY")

    # Logging settings
      log_code: bool = True
    log_graph: bool = True
    log_git_metadata: bool = True
    log_git_patch: bool = True
    auto_metric_logging: bool = True
    auto_param_logging: bool = True

    # What to log
    log_confusion_matrix: bool = True
    log_images: bool = True
    log_model: bool = True
    log_metrics_every_n_batches: int = 10

    # Tags and metadata
    tags: List[str] = field(default_factory=lambda: [
              "vision-transformer",
              "agriculture",
              "disease-detection"
    ])


@dataclass
class BERTConfig:
      """Configuration for BERT recommendation engine"""
      # Model settings
      model_name: str = "bert-base-uncased"
    max_length: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9

    # Recommendation settings
    generate_recommendation: bool = True
    recommendation_template: str = (
              "Based on the detected {disease} in {crop}, "
              "we recommend the following actions: {actions}"
    )


@dataclass
class DeploymentConfig:
      """Configuration for deployment"""
      # API settings
      api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False

    # Model serving
    model_path: str = "models/best_model.pt"
    device: str = "cpu"  # Will use GPU if available

    # Request handling
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    timeout: int = 30  # seconds


@dataclass
class ExplainabilityConfig:
      """Configuration for explainable AI features"""
      # Grad-CAM settings
      enable_gradcam: bool = True
    target_layer: Optional[str] = None  # Will use last conv layer if None

    # Visualization
    save_heatmaps: bool = True
    heatmap_alpha: float = 0.4
    colormap: str = "jet"


class Config:
      """Master configuration class combining all configs"""

    def __init__(self):
              self.paths = PathConfig()
              self.model = ModelConfig()
              self.training = TrainingConfig()
              self.data = DataConfig()
              self.comet = CometConfig()
              self.bert = BERTConfig()
              self.deployment = DeploymentConfig()
              self.explainability = ExplainabilityConfig()

        # Create necessary directories
              self.paths.create_directories()

    def update_num_classes(self, num_classes: int):
              """Update number of classes based on dataset"""
              self.model.num_classes = num_classes

    def to_dict(self) -> Dict:
              """Convert configuration to dictionary"""
              return {
                  'paths': self.paths.__dict__,
                  'model': self.model.__dict__,
                  'training': self.training.__dict__,
                  'data': self.data.__dict__,
                  'comet': self.comet.__dict__,
                  'bert': self.bert.__dict__,
                  'deployment': self.deployment.__dict__,
                  'explainability': self.explainability.__dict__
              }

    def __repr__(self) -> str:
              """String representation of configuration"""
              config_dict = self.to_dict()
              lines = ["Configuration:"]
              for section, params in config_dict.items():
                            lines.append(f"\n{section.upper()}:")
                            for key, value in params.items():
                                              lines.append(f"  {key}: {value}")
                                      return "\n".join(lines)


# Global configuration instance
config = Config()


if __name__ == "__main__":
      # Print configuration
      print(config)

    # Example: Update number of classes
      config.update_num_classes(12)
      print(f"\nUpdated num_classes: {config.model.num_classes}")
