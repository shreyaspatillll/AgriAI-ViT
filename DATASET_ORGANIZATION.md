# Dataset Organization Guide for AgriAI-ViT

## Overview
This document describes the dataset structure and organization for the AgriAI-ViT project.

## Source Datasets

### Location
- **Original Location**: `/Users/shreyaspatil/Downloads/Project/Dataset/`
- **Project Location**: `/Users/shreyaspatil/AgriAI-ViT/data/raw/`

### Dataset Structure

#### Rice Disease Dataset
Located at: `Rice Disease/`

**Classes (8 total):**
1. Bacterial Leaf Blight
2. Brown Spot
3. Healthy Rice Leaf
4. Insect
5. Leaf Blast
6. Leaf Scald
7. Rice Tungro
8. Sheath Blight

**Note**: For the initial AgriAI-ViT model, we will focus on these 4 main classes:
- Rice_Bacterial_Leaf_Blight
- Rice_Brown_Spot
- Rice_Leaf_Blast
- Rice_Healthy

#### Wheat Disease Dataset
Located at: `Wheat Disease/`

**Structure**: Pre-split into train/valid/test folders

**Classes (4 expected):**
- Wheat_Brown_Rust
- Wheat_Yellow_Rust
- Wheat_Septoria
- Wheat_Healthy

## Target Directory Structure

After running the data preparation script, the data will be organized as:

```
data/
├── raw/
│   ├── rice/
│   │   ├── Rice_Bacterial_Leaf_Blight/
│   │   ├── Rice_Brown_Spot/
│   │   ├── Rice_Leaf_Blast/
│   │   └── Rice_Healthy/
│   └── wheat/
│       ├── Wheat_Brown_Rust/
│       ├── Wheat_Yellow_Rust/
│       ├── Wheat_Septoria/
│       └── Wheat_Healthy/
└── processed/
    ├── train/
    │   ├── Rice_Bacterial_Leaf_Blight/
    │   ├── Rice_Brown_Spot/
    │   ├── Rice_Leaf_Blast/
    │   ├── Rice_Healthy/
    │   ├── Wheat_Brown_Rust/
    │   ├── Wheat_Yellow_Rust/
    │   ├── Wheat_Septoria/
    │   └── Wheat_Healthy/
    ├── val/
    │   └── [same classes]
    └── test/
        └── [same classes]
```

## Data Split Configuration

- **Training**: 70%
- **Validation**: 20%
- **Testing**: 10%

## Data Preparation Steps

### Step 1: Copy Raw Data
```bash
# Run the data preparation script
python src/data_preparation.py
```

This will:
1. Copy rice disease images from source to `data/raw/rice/`
2. Copy wheat disease images from source to `data/raw/wheat/`
3. Organize images by class

### Step 2: Preprocess and Split
The script will automatically:
1. Resize images to 224×224
2. Split data into train/val/test sets
3. Apply data augmentation (training set only)
4. Save processed data to `data/processed/`

### Step 3: Create Metadata
Generates:
- `data/processed/class_mapping.json` - Maps class indices to names
- `data/processed/dataset_statistics.json` - Dataset statistics
- `data/processed/split_info.json` - Information about the data split

## Manual Dataset Organization (Alternative)

If you prefer to organize datasets manually:

### For Rice Disease:
```bash
# Create directories
mkdir -p data/raw/rice/{Rice_Bacterial_Leaf_Blight,Rice_Brown_Spot,Rice_Leaf_Blast,Rice_Healthy}

# Copy images
cp -r "/Users/shreyaspatil/Downloads/Project/Dataset/Rice Disease/Bacterial Leaf Blight/"* data/raw/rice/Rice_Bacterial_Leaf_Blight/
cp -r "/Users/shreyaspatil/Downloads/Project/Dataset/Rice Disease/Brown Spot/"* data/raw/rice/Rice_Brown_Spot/
cp -r "/Users/shreyaspatil/Downloads/Project/Dataset/Rice Disease/Leaf Blast/"* data/raw/rice/Rice_Leaf_Blast/
cp -r "/Users/shreyaspatil/Downloads/Project/Dataset/Rice Disease/Healthy Rice Leaf/"* data/raw/rice/Rice_Healthy/
```

### For Wheat Disease:
```bash
# Copy wheat dataset (already split)
cp -r "/Users/shreyaspatil/Downloads/Project/Dataset/Wheat Disease/" data/raw/wheat/
```

## Class Naming Convention

All class names follow the pattern: `{Crop}_{Disease}`

- **Rice classes**: Rice_Bacterial_Leaf_Blight, Rice_Brown_Spot, Rice_Leaf_Blast, Rice_Healthy
- **Wheat classes**: Wheat_Brown_Rust, Wheat_Yellow_Rust, Wheat_Septoria, Wheat_Healthy

## Total Classes: 8

1. Rice_Bacterial_Leaf_Blight
2. Rice_Brown_Spot
3. Rice_Leaf_Blast
4. Rice_Healthy
5. Wheat_Brown_Rust
6. Wheat_Yellow_Rust
7. Wheat_Septoria
8. Wheat_Healthy

## Image Requirements

- **Format**: JPG, JPEG, PNG
- **Input Size**: Variable (will be resized to 224×224)
- **Color**: RGB (3 channels)
- **Quality**: High resolution leaf images

## Data Augmentation (Training Only)

Applied transformations:
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Random crop with padding
- Color jitter (brightness, contrast, saturation)
- Normalization (ImageNet statistics)

## Validation and Test Sets

- **No augmentation** applied
- Only resize and normalization
- Ensures consistent evaluation

## Dataset Statistics

Run after data preparation:
```python
python -c "from src.data_preparation import analyze_dataset; analyze_dataset('data/processed')"
```

This will display:
- Total images per class
- Train/val/test split counts
- Image size distribution
- Class balance information

## Troubleshooting

### Issue: Dataset not found
- Verify source path: `/Users/shreyaspatil/Downloads/Project/Dataset/`
- Check permissions on source and target directories

### Issue: Insufficient space
- Rice + Wheat datasets may require 2-5 GB
- Ensure adequate disk space before copying

### Issue: Class mismatch
- Verify all 8 classes are present
- Check class naming matches config.py

## Next Steps

After dataset organization:
1. Verify class counts: `python src/data_preparation.py --verify`
2. Train model: `python src/train.py`
3. Evaluate: `python src/evaluate.py`

---

**Last Updated**: January 2025  
**Maintained by**: AgriAI-ViT Team
