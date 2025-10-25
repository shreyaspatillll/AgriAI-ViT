# AgriAI-ViT Implementation Guide

## Table of Contents
- [System Requirements](#system-requirements)
- [Installation Steps](#installation-steps)
- [Dataset Setup](#dataset-setup)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Running Data Pipeline](#running-data-pipeline)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [API Deployment](#api-deployment)
- [Testing](#testing)

---

## System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA GPU 8GB+ VRAM | RTX 3080/3090 10GB+ |
| **RAM** | 16GB DDR4 | 32GB DDR4 |
| **Storage** | 50GB SSD | 100GB+ SSD |
| **CPU** | Intel i5/Ryzen 5 | Intel i7/Ryzen 7 |

### Software Requirements

- **OS:** Ubuntu 20.04+ / Windows 10+ / macOS 11+
- **Python:** 3.8, 3.9, or 3.10
- **CUDA:** 11.3+ (for GPU)
- **Git:** 2.25+
- **Docker:** 20.10+ (optional)

---

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/shreyaspatillll/AgriAI-ViT.git
cd AgriAI-ViT
```

### 2. Create Virtual Environment

```bash
python3 -m venv agriai_env
source agriai_env/bin/activate  # Linux/macOS
# agriai_env\Scripts\activate  # Windows
```

### 3. Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Dataset Setup

### Download from Kaggle

```bash
# Install Kaggle CLI
pip install kaggle

# Setup credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download datasets
cd data/raw
kaggle datasets download -d vipoooool/new-plant-diseases-dataset --unzip
```

### Organize Dataset Structure

```bash
python scripts/prepare_dataset.py \
  --input_dir data/raw \
  --output_dir data/processed \
  --train_split 0.70 \
  --val_split 0.15 \
  --test_split 0.15
```

Expected structure:
```
data/
├── raw/                 # Original data
├── processed/
│   ├── train/          # 70%
│   ├── val/            # 15%
│   └── test/           # 15%
└── metadata/
    └── class_mapping.json
```

---

## Configuration

### Environment Variables

Create `.env` file:

```bash
# Comet ML
COMET_API_KEY=your_api_key_here
COMET_PROJECT_NAME=agriai-vit
COMET_WORKSPACE=your_workspace

# Paths
DATA_DIR=data/processed
MODEL_DIR=models/checkpoints

# Training
CUDA_VISIBLE_DEVICES=0
```

### Model Configuration

Edit `config/model_config.yaml`:

```yaml
model:
  name: "vit_base_patch16_224"
  pretrained: true
  num_classes: 38

training:
  batch_size: 32
  epochs: 30
  learning_rate: 3e-5
  weight_decay: 0.01
  early_stopping_patience: 5

optimizer:
  name: "AdamW"
  betas: [0.9, 0.999]
```

---

## Running Data Pipeline

### Preprocessing

```bash
python src/data/preprocessing.py \
  --data_dir data/processed \
  --resize 224 \
  --normalize
```

### Data Augmentation

```bash
python src/data/augmentation.py \
  --input_dir data/processed/train \
  --augment_factor 2
```

### Create DataLoaders

```python
from src.data.dataloader import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    data_dir='data/processed',
    batch_size=32,
    num_workers=4
)
```

---

## Model Training

### Single Model Training

```bash
python scripts/train.py \
  --config config/model_config.yaml \
  --experiment_name "vit-base-run1" \
  --log_comet
```

### Training with Comet ML Logging

```python
from comet_ml import Experiment
from src.models.vit_model import ViTModel

# Initialize experiment
experiment = Experiment(
    api_key=os.getenv('COMET_API_KEY'),
    project_name='agriai-vit'
)

# Train model
model = ViTModel(num_classes=38)
trainer = Trainer(model, experiment)
trainer.train(train_loader, val_loader, epochs=30)
```

### Monitor Training

- Visit Comet ML dashboard
- Track metrics: accuracy, loss, F1-score
- View confusion matrices and training curves

---

## Evaluation

### Run Evaluation Script

```bash
python scripts/evaluate.py \
  --model_path models/checkpoints/best_model.pt \
  --test_data data/processed/test \
  --output_dir results/
```

### Generate Metrics

```python
from src.evaluation.metrics import calculate_metrics

metrics = calculate_metrics(
    y_true=test_labels,
    y_pred=predictions,
    class_names=class_names
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
```

---

## API Deployment

### Start FastAPI Server

```bash
cd deployment
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment

```bash
# Build image
docker build -t agriai-api:latest .

# Run container
docker run -p 8000:8000 agriai-api:latest
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@sample_leaf.jpg"
```

---

## Testing

### Run Unit Tests

```bash
pytest tests/ -v --cov=src
```

### Test Individual Components

```bash
# Test data pipeline
pytest tests/test_dataloader.py

# Test model
pytest tests/test_model.py

# Test API
pytest tests/test_api.py
```

---

## Quick Start Commands

```bash
# Complete setup
make setup

# Prepare data
make data

# Train model
make train

# Deploy API
make deploy

# Run all tests
make test
```

---

## Troubleshooting

### CUDA Out of Memory
- Reduce batch_size in config
- Use gradient accumulation
- Enable mixed precision training

### Slow Training
- Check GPU utilization: `nvidia-smi`
- Increase num_workers in DataLoader
- Use larger batch size if memory allows

### API Connection Issues
- Check firewall settings
- Verify port 8000 is not in use
- Check Docker network settings

---

## Additional Resources

- [Comet ML Documentation](https://www.comet.ml/docs/)
- [Vision Transformers Paper](https://arxiv.org/abs/2010.11929)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**For detailed training guide, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md)**
**For API usage, see [API_USAGE_GUIDE.md](API_USAGE_GUIDE.md)**
