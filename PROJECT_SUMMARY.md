# AgriAI-ViT Project Summary

## 📋 Project Overview

**Project Name**: AgriAI-ViT (Vision Transformer-Powered AI Recommendations for Smart Agriculture in India)

**Version**: 1.0.0

**Location**: `/Users/shreyaspatil/AgriAI-ViT/`

**GitHub Repository**: https://github.com/shreyaspatillll/AgriAI-ViT

## 🎯 Project Objectives

Develop an end-to-end AI system for smart agriculture that:
- Detects crop diseases, pests, and weeds in wheat and rice crops using Vision Transformers (ViT)
- Generates actionable, localized recommendations using BERT-based text generation
- Tracks and visualizes experiments through Comet ML
- Deploys real-time inference via FastAPI and Bolt.AI
- Helps optimize crop yield and farmer income across India

## 🗂️ Project Structure

```
AgriAI-ViT/
├── configs/
│   └── config.py                    # Configuration settings
├── data/
│   ├── raw/                         # Raw datasets (rice & wheat)
│   └── processed/                   # Preprocessed and split data
├── deployment/
│   ├── app.py                       # FastAPI application
│   ├── Dockerfile                   # Docker configuration
│   └── docker-compose.yml           # Docker Compose setup
├── logs/                            # Application logs
├── models/                          # Saved model checkpoints
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Data analysis
│   ├── 02_model_training.ipynb      # Training demo
│   ├── 03_evaluation.ipynb          # Evaluation demo
│   └── 04_inference_demo.ipynb      # Inference demo
├── scripts/
│   ├── organize_datasets.sh         # Bash dataset organizer
│   └── organize_datasets.py         # Python dataset organizer
├── src/
│   ├── __init__.py
│   ├── data_preparation.py          # Dataset preprocessing
│   ├── dataset.py                   # PyTorch dataset classes
│   ├── vit_model.py                 # Vision Transformer model
│   ├── train.py                     # Training script
│   ├── bert_recommendations.py      # BERT recommendation engine
│   ├── inference.py                 # Inference utilities
│   └── evaluate.py                  # Evaluation script
├── tests/
│   ├── __init__.py
│   ├── test_model.py                # ViT model tests
│   ├── test_recommendations.py      # Recommendation tests
│   └── test_api.py                  # API endpoint tests
├── utils/
│   ├── __init__.py
│   ├── comet_logger.py              # Comet ML logging
│   └── visualizations.py            # Plotting utilities
├── .env.example                     # Environment variables template
├── .gitignore                       # Git ignore rules
├── CONTRIBUTING.md                  # Contribution guidelines
├── DATASET_ORGANIZATION.md          # Dataset structure guide
├── DEPLOYMENT.md                    # Deployment instructions
├── LICENSE                          # MIT License
├── Makefile                         # Build automation
├── PROJECT_SUMMARY.md               # This file
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
└── setup.py                         # Package setup
```

## 🔬 Technical Architecture

### 1. **Vision Transformer (ViT) Model**
- **Model**: `vit_base_patch16_224` (timm library)
- **Pretrained**: ImageNet-21k weights
- **Classes**: 8 (4 rice + 4 wheat diseases)
- **Input Size**: 224×224 RGB images
- **Framework**: PyTorch with mixed precision training

### 2. **BERT Recommendation Engine**
- **Model**: `bert-base-uncased` (Hugging Face)
- **Purpose**: Generate localized treatment recommendations
- **Knowledge Base**: Research-backed disease management protocols
- **Output**: Symptoms, treatments, preventive measures, organic alternatives

### 3. **Dataset**
- **Rice Classes** (4):
  - Rice_Bacterial_Leaf_Blight
  - Rice_Brown_Spot
  - Rice_Leaf_Blast
  - Rice_Healthy
  
- **Wheat Classes** (4):
  - Wheat_Brown_Rust
  - Wheat_Yellow_Rust
  - Wheat_Septoria
  - Wheat_Healthy

- **Split**: 70% train, 20% validation, 10% test

### 4. **Training Configuration**
- **Optimizer**: AdamW (lr=3e-5, weight_decay=0.01)
- **Loss**: CrossEntropyLoss
- **Batch Size**: 32
- **Epochs**: 30 (with EarlyStopping, patience=5)
- **Scheduler**: ReduceLROnPlateau
- **Device**: CUDA with mixed precision (AMP)

### 5. **Experiment Tracking**
- **Platform**: Comet ML
- **Project**: "AgriAI-ViT"
- **Logged**: Hyperparameters, metrics, confusion matrices, ROC curves, model checkpoints

### 6. **Deployment**
- **Framework**: FastAPI
- **Endpoints**:
  - `POST /predict`: Image upload → disease + recommendation
  - `GET /health`: API health check
  - `GET /classes`: List available classes
- **Containerization**: Docker + Docker Compose
- **Cloud**: Ready for Render, Heroku, AWS, GCP

## 📦 Dependencies

### Core Libraries
- **PyTorch** (2.0+): Deep learning framework
- **timm**: Vision Transformer models
- **transformers**: BERT models (Hugging Face)
- **torchvision**: Image transformations
- **comet-ml**: Experiment tracking

### Data & Visualization
- **pandas**, **numpy**: Data manipulation
- **matplotlib**, **seaborn**: Plotting
- **Pillow**: Image processing
- **albumentations**: Advanced augmentation

### API & Deployment
- **FastAPI**: REST API framework
- **uvicorn**: ASGI server
- **python-multipart**: File upload support

### Testing
- **pytest**: Unit testing
- **pytest-cov**: Code coverage

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/shreyaspatillll/AgriAI-ViT.git
cd AgriAI-ViT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env with your Comet API key
```

### 2. Dataset Organization
```bash
# Option 1: Shell script
chmod +x scripts/organize_datasets.sh
./scripts/organize_datasets.sh

# Option 2: Python script
python scripts/organize_datasets.py
```

### 3. Data Preprocessing
```bash
python src/data_preparation.py
```

### 4. Model Training
```bash
python src/train.py
```

### 5. Model Evaluation
```bash
python src/evaluate.py
```

### 6. API Deployment
```bash
# Local
uvicorn deployment.app:app --reload

# Docker
docker-compose -f deployment/docker-compose.yml up
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_model.py -v
```

## 📊 Evaluation Metrics

The model is evaluated on:
- **Accuracy**: Overall and per-class
- **Precision**: Disease detection precision
- **Recall**: Disease detection sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Class-wise predictions
- **ROC Curves**: Receiver Operating Characteristic
- **Inference Time**: Per-image prediction speed

## 🎨 Explainable AI

- **Grad-CAM Visualizations**: Highlight disease-affected regions
- **Attention Maps**: ViT attention visualization
- **Confidence Scores**: Prediction certainty
- **Logged to Comet**: Visual artifacts for analysis

## 📈 Expected Outcomes

1. **High Accuracy**: >90% on test set
2. **Real-time Inference**: <100ms per image
3. **Scalable Deployment**: Docker + cloud-ready
4. **Explainable Predictions**: Grad-CAM heatmaps
5. **Actionable Recommendations**: Research-backed treatments

## 🔗 Key Resources

- **GitHub**: https://github.com/shreyaspatillll/AgriAI-ViT
- **Comet ML**: Set COMET_API_KEY in .env
- **Documentation**: README.md, DEPLOYMENT.md, DATASET_ORGANIZATION.md
- **Model Checkpoints**: Saved in `models/` directory
- **Logs**: Available in `logs/` directory

## 🛠️ Development Tools

### Makefile Commands
```bash
make install        # Install dependencies
make setup          # Full project setup
make data           # Prepare datasets
make train          # Train model
make evaluate       # Evaluate model
make deploy         # Deploy API
make test           # Run tests
make clean          # Clean temporary files
make docker-build   # Build Docker image
make docker-run     # Run Docker container
```

## 📝 Configuration

All settings are centralized in `configs/config.py`:
- Model architecture
- Training hyperparameters
- Data paths
- Augmentation settings
- Device configuration

Environment variables in `.env`:
- Comet ML credentials
- Kaggle API keys
- API configuration

## 🤝 Contributing

See `CONTRIBUTING.md` for guidelines on:
- Code style
- Pull request process
- Issue reporting
- Documentation standards

## 📄 License

This project is licensed under the MIT License. See `LICENSE` file for details.

## 👥 Team

**Maintainer**: AgriAI Team  
**Contact**: shreyaspatillll (GitHub)

## 🎓 Academic Context

This project demonstrates:
- **Transfer Learning**: ImageNet → Agricultural domains
- **Multi-modal AI**: Vision (ViT) + Language (BERT)
- **MLOps Best Practices**: Comet ML tracking, Docker deployment
- **Domain-Specific AI**: Tailored for Indian agriculture

## 🌾 Impact

**Target Audience**: Indian farmers growing wheat and rice

**Benefits**:
- Early disease detection
- Reduced crop loss
- Optimized treatment application
- Increased yield and income
- Accessible AI technology

## 📅 Project Timeline

- **Phase 1**: Dataset organization ✅
- **Phase 2**: Model training (In Progress)
- **Phase 3**: Evaluation & optimization
- **Phase 4**: Deployment & testing
- **Phase 5**: Production release

## 🔮 Future Enhancements

1. **More Crops**: Extend to maize, cotton, sugarcane
2. **Mobile App**: Android/iOS deployment
3. **Regional Languages**: Hindi, Tamil, Telugu support
4. **Weather Integration**: Location-based recommendations
5. **Drone Integration**: Automated field scanning
6. **IoT Sensors**: Real-time crop monitoring

---

**Last Updated**: January 25, 2025  
**Status**: Development - Dataset Organization Complete  
**Next Milestone**: Model Training
