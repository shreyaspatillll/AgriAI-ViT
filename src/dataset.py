"""
AgriAI-ViT Dataset Module
Handles data loading, preprocessing, and augmentation for rice and wheat disease detection
"""

import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json


class AgriDataset(Dataset):
      """Custom dataset for agricultural disease detection"""

    def __init__(
              self,
              data_dir: str,
              split: str = 'train',
              transform: Optional[transforms.Compose] = None,
              img_size: int = 224
    ):
              """
                      Initialize the dataset

                                      Args:
                                                  data_dir: Root directory containing train/val/test splits
                                                              split: One of 'train', 'val', or 'test'
                                                                          transform: Optional transform to be applied on images
                                                                                      img_size: Target image size (default: 224 for ViT)
                                                                                              """
              self.data_dir = Path(data_dir)
              self.split = split
              self.img_size = img_size

        # Set up transforms
              if transform is None:
                            self.transform = self._get_default_transform()
else:
            self.transform = transform

          # Load image paths and labels
          self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self._load_dataset()

    def _get_default_transform(self) -> transforms.Compose:
              """Get default transforms based on split"""
              if self.split == 'train':
                            return transforms.Compose([
                                              transforms.Resize((self.img_size, self.img_size)),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomRotation(degrees=15),
                                              transforms.RandomCrop(self.img_size, padding=4),
                                              transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                              transforms.ToTensor(),
                                              transforms.Normalize(
                                                                    mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225]
                                              )
                            ])
else:
            return transforms.Compose([
                              transforms.Resize((self.img_size, self.img_size)),
                              transforms.ToTensor(),
                              transforms.Normalize(
                                                    mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]
                              )
            ])

    def _load_dataset(self):
              """Load dataset from directory structure"""
              split_dir = self.data_dir / self.split

        if not split_dir.exists():
                      raise ValueError(f"Split directory {split_dir} does not exist")

        # Get all class directories
        class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        # Create class mappings
        self.class_to_idx = {cls_dir.name: idx for idx, cls_dir in enumerate(class_dirs)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

        # Load all images
        for class_dir in class_dirs:
                      class_name = class_dir.name
                      class_idx = self.class_to_idx[class_name]

            # Get all image files
                      image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
                      image_files = [
                          f for f in class_dir.iterdir()
                          if f.suffix in image_extensions
                      ]

            for img_path in image_files:
                              self.samples.append((str(img_path), class_idx))

        print(f"Loaded {len(self.samples)} images from {len(self.class_to_idx)} classes for {self.split} split")

    def __len__(self) -> int:
              return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
              """Get a sample from the dataset"""
              img_path, label = self.samples[idx]

        # Load image
              image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
                      image = self.transform(image)

        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
              """Get distribution of samples across classes"""
              distribution = {}
              for _, label in self.samples:
                            class_name = self.idx_to_class[label]
                            distribution[class_name] = distribution.get(class_name, 0) + 1
                        return distribution


def get_data_loaders(
      data_dir: str,
      batch_size: int = 32,
      num_workers: int = 4,
      img_size: int = 224,
      pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
      """
          Create data loaders for train, validation, and test sets

                  Args:
                          data_dir: Root directory containing train/val/test splits
                                  batch_size: Batch size for data loaders
                                          num_workers: Number of worker processes for data loading
                                                  img_size: Target image size
                                                          pin_memory: Whether to pin memory for faster GPU transfer

                                                                      Returns:
                                                                              Tuple of (train_loader, val_loader, test_loader)
                                                                                  """
    # Create datasets
    train_dataset = AgriDataset(data_dir, split='train', img_size=img_size)
    val_dataset = AgriDataset(data_dir, split='val', img_size=img_size)
    test_dataset = AgriDataset(data_dir, split='test', img_size=img_size)

    # Create data loaders
    train_loader = DataLoader(
              train_dataset,
              batch_size=batch_size,
              shuffle=True,
              num_workers=num_workers,
              pin_memory=pin_memory,
              drop_last=True
    )

    val_loader = DataLoader(
              val_dataset,
              batch_size=batch_size,
              shuffle=False,
              num_workers=num_workers,
              pin_memory=pin_memory
    )

    test_loader = DataLoader(
              test_dataset,
              batch_size=batch_size,
              shuffle=False,
              num_workers=num_workers,
              pin_memory=pin_memory
    )

    # Print dataset info
    print(f"\nDataset Statistics:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")
    print(f"Number of classes: {len(train_dataset.class_to_idx)}")
    print(f"Classes: {list(train_dataset.class_to_idx.keys())}")

    return train_loader, val_loader, test_loader


def get_class_weights(data_dir: str, split: str = 'train') -> torch.Tensor:
      """
          Calculate class weights for handling class imbalance

                  Args:
                          data_dir: Root directory containing data
                                  split: Dataset split to calculate weights from

                                              Returns:
                                                      Tensor of class weights
                                                          """
    dataset = AgriDataset(data_dir, split=split)
    distribution = dataset.get_class_distribution()

    total_samples = sum(distribution.values())
    num_classes = len(distribution)

    weights = []
    for class_name in sorted(distribution.keys()):
              count = distribution[class_name]
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
      # Test the dataset
      data_dir = "data/processed"

    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
              data_dir=data_dir,
              batch_size=32,
              num_workers=4
    )

    # Test batch loading
    for images, labels in train_loader:
              print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        break

    # Print class distribution
    train_dataset = AgriDataset(data_dir, split='train')
    print("\nClass Distribution:")
    for class_name, count in train_dataset.get_class_distribution().items():
              print(f"{class_name}: {count}")

    # Calculate class weights
    weights = get_class_weights(data_dir)
    print(f"\nClass weights: {weights}")
