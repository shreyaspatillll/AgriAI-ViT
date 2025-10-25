# 📚 AgriAI-ViT Project Index

**Quick Navigation Guide for All Project Files**

---

## 🚀 START HERE

1. **[TASK_COMPLETION_SUMMARY.md](TASK_COMPLETION_SUMMARY.md)** - Complete task summary and status
2. **[SETUP_VERIFICATION.md](SETUP_VERIFICATION.md)** - Setup checklist and verification steps
3. **[README.md](README.md)** - Project overview and quick start
4. **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Comprehensive project details

---

## 📖 Documentation Files

### Getting Started
- **[README.md](README.md)** - Main project documentation
- **[SETUP_VERIFICATION.md](SETUP_VERIFICATION.md)** - Setup and verification guide
- **[TASK_COMPLETION_SUMMARY.md](TASK_COMPLETION_SUMMARY.md)** - Task completion report

### Technical Guides
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment instructions (local, Docker, cloud)
- **[DATASET_ORGANIZATION.md](DATASET_ORGANIZATION.md)** - Dataset structure and organization
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Detailed project overview
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

### Legacy Documentation (Previous iterations)
- **[FILE_INVENTORY.md](FILE_INVENTORY.md)** - File inventory from previous task
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Project status from previous task
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide
- **[START_HERE.md](START_HERE.md)** - Alternative start guide
- **[SETUP_CHECKLIST.md](SETUP_CHECKLIST.md)** - Setup checklist
- **[TASK_COMPLETION_REPORT.md](TASK_COMPLETION_REPORT.md)** - Previous completion report
- **[FINAL_SUMMARY.txt](FINAL_SUMMARY.txt)** - Previous summary
- **[README_FIRST.txt](README_FIRST.txt)** - Previous start guide
- **[USER_ACTION_REQUIRED.txt](USER_ACTION_REQUIRED.txt)** - Previous action items

---

## 🐍 Source Code (src/)

### Core Components
- **[src/vit_model.py](src/vit_model.py)** - Vision Transformer model implementation
- **[src/bert_recommendations.py](src/bert_recommendations.py)** - BERT recommendation engine with disease mappings
- **[src/train.py](src/train.py)** - Training script with Comet ML integration
- **[src/evaluate.py](src/evaluate.py)** - Model evaluation script
- **[src/inference.py](src/inference.py)** - Inference utilities

### Data Processing
- **[src/data_preparation.py](src/data_preparation.py)** - Dataset preprocessing and splitting
- **[src/dataset.py](src/dataset.py)** - PyTorch dataset classes with augmentation

### Package Files
- **[src/__init__.py](src/__init__.py)** - Source package initialization

---

## 🔧 Utility Code (utils/)

- **[utils/comet_logger.py](utils/comet_logger.py)** - Comet ML experiment tracking wrapper
- **[utils/visualizations.py](utils/visualizations.py)** - Plotting and visualization utilities
- **[utils/__init__.py](utils/__init__.py)** - Utils package initialization

---

## 🚀 Deployment (deployment/)

- **[deployment/app.py](deployment/app.py)** - FastAPI application
- **[deployment/Dockerfile](deployment/Dockerfile)** - Docker configuration
- **[deployment/docker-compose.yml](deployment/docker-compose.yml)** - Docker Compose setup

---

## 🧪 Tests (tests/)

- **[tests/test_model.py](tests/test_model.py)** - ViT model unit tests
- **[tests/test_recommendations.py](tests/test_recommendations.py)** - Recommendation engine tests
- **[tests/test_api.py](tests/test_api.py)** - API endpoint tests
- **[tests/__init__.py](tests/__init__.py)** - Tests package initialization

---

## 📓 Notebooks (notebooks/)

- **[notebooks/01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb)** - Data analysis
- **[notebooks/02_model_training.ipynb](notebooks/02_model_training.ipynb)** - Training demo
- **[notebooks/03_evaluation.ipynb](notebooks/03_evaluation.ipynb)** - Evaluation demo
- **[notebooks/04_inference_demo.ipynb](notebooks/04_inference_demo.ipynb)** - Inference demo

---

## 📜 Scripts (scripts/)

- **[scripts/organize_datasets.py](scripts/organize_datasets.py)** - Python dataset organization script
- **[scripts/organize_datasets.sh](scripts/organize_datasets.sh)** - Bash dataset organization script

---

## ⚙️ Configuration Files

### Main Configuration
- **[configs/config.py](configs/config.py)** - Main configuration settings

### Project Configuration
- **[requirements.txt](requirements.txt)** - Python dependencies
- **[setup.py](setup.py)** - Package setup configuration
- **[Makefile](Makefile)** - Build automation commands

### Environment
- **[.env.example](.env.example)** - Environment variables template
- **[.gitignore](.gitignore)** - Git ignore rules

### Legal
- **[LICENSE](LICENSE)** - MIT License

---

## 📁 Directory Structure

### Data Directories
- **data/raw/** - Raw dataset storage (rice & wheat)
  - **[data/raw/.gitkeep](data/raw/.gitkeep)** - Placeholder
- **data/processed/** - Preprocessed and split data
  - **[data/processed/.gitkeep](data/processed/.gitkeep)** - Placeholder

### Output Directories
- **models/** - Saved model checkpoints
  - **[models/.gitkeep](models/.gitkeep)** - Placeholder
- **logs/** - Application logs
  - **[logs/.gitkeep](logs/.gitkeep)** - Placeholder

---

## 🎯 Quick Reference

### For First-Time Setup
1. Read [TASK_COMPLETION_SUMMARY.md](TASK_COMPLETION_SUMMARY.md)
2. Follow [SETUP_VERIFICATION.md](SETUP_VERIFICATION.md)
3. Run `python scripts/organize_datasets.py`

### For Development
1. Source code: [src/](src/)
2. Tests: [tests/](tests/)
3. Configuration: [configs/config.py](configs/config.py)
4. Utilities: [utils/](utils/)

### For Deployment
1. API: [deployment/app.py](deployment/app.py)
2. Docker: [deployment/Dockerfile](deployment/Dockerfile)
3. Guide: [DEPLOYMENT.md](DEPLOYMENT.md)

### For Data
1. Organization guide: [DATASET_ORGANIZATION.md](DATASET_ORGANIZATION.md)
2. Preparation script: [src/data_preparation.py](src/data_preparation.py)
3. Dataset classes: [src/dataset.py](src/dataset.py)

### For Training
1. Training script: [src/train.py](src/train.py)
2. Model definition: [src/vit_model.py](src/vit_model.py)
3. Configuration: [configs/config.py](configs/config.py)

### For Evaluation
1. Evaluation script: [src/evaluate.py](src/evaluate.py)
2. Visualizations: [utils/visualizations.py](utils/visualizations.py)
3. Comet logging: [utils/comet_logger.py](utils/comet_logger.py)

### For Inference
1. Inference utilities: [src/inference.py](src/inference.py)
2. Recommendations: [src/bert_recommendations.py](src/bert_recommendations.py)
3. API: [deployment/app.py](deployment/app.py)

---

## 📊 File Statistics

| Category | Count | Description |
|----------|-------|-------------|
| **Documentation** | 15+ files | Markdown guides and READMEs |
| **Source Code** | 8 files | Core Python modules |
| **Utilities** | 3 files | Helper functions |
| **Tests** | 4 files | Unit tests |
| **Notebooks** | 4 files | Jupyter notebooks |
| **Scripts** | 2 files | Dataset organization |
| **Deployment** | 3 files | API and Docker |
| **Configuration** | 6 files | Settings and dependencies |
| **Total** | 40+ files | Complete project |

---

## 🔍 Search by Purpose

### Want to understand the project?
→ Start with [README.md](README.md) and [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

### Want to set up the project?
→ Follow [SETUP_VERIFICATION.md](SETUP_VERIFICATION.md)

### Want to organize datasets?
→ Read [DATASET_ORGANIZATION.md](DATASET_ORGANIZATION.md), run [scripts/organize_datasets.py](scripts/organize_datasets.py)

### Want to train the model?
→ Configure [configs/config.py](configs/config.py), run [src/train.py](src/train.py)

### Want to evaluate results?
→ Run [src/evaluate.py](src/evaluate.py), check [notebooks/03_evaluation.ipynb](notebooks/03_evaluation.ipynb)

### Want to deploy the API?
→ Follow [DEPLOYMENT.md](DEPLOYMENT.md), use [deployment/app.py](deployment/app.py)

### Want to run tests?
→ Execute `pytest tests/` or check individual test files in [tests/](tests/)

### Want to understand disease recommendations?
→ Review [src/bert_recommendations.py](src/bert_recommendations.py)

### Want to visualize results?
→ Use [utils/visualizations.py](utils/visualizations.py) or [notebooks/](notebooks/)

---

## 🎯 Common Tasks

### Task 1: Organize Datasets
```bash
python scripts/organize_datasets.py
```
**Related Files**:
- [scripts/organize_datasets.py](scripts/organize_datasets.py)
- [DATASET_ORGANIZATION.md](DATASET_ORGANIZATION.md)

### Task 2: Install Dependencies
```bash
pip install -r requirements.txt
```
**Related Files**:
- [requirements.txt](requirements.txt)
- [setup.py](setup.py)

### Task 3: Train Model
```bash
python src/train.py
```
**Related Files**:
- [src/train.py](src/train.py)
- [src/vit_model.py](src/vit_model.py)
- [configs/config.py](configs/config.py)

### Task 4: Run API
```bash
uvicorn deployment.app:app --reload
```
**Related Files**:
- [deployment/app.py](deployment/app.py)
- [DEPLOYMENT.md](DEPLOYMENT.md)

### Task 5: Run Tests
```bash
pytest tests/ -v
```
**Related Files**:
- [tests/test_model.py](tests/test_model.py)
- [tests/test_recommendations.py](tests/test_recommendations.py)
- [tests/test_api.py](tests/test_api.py)

---

## 🌟 Key Features by File

### ViT Model ([src/vit_model.py](src/vit_model.py))
- Vision Transformer implementation
- Pretrained weights loading
- Feature extraction mode
- Freeze/unfreeze capability

### BERT Recommendations ([src/bert_recommendations.py](src/bert_recommendations.py))
- 8 disease class mappings
- Research-based treatments
- Symptoms and preventive measures
- Organic alternatives

### Training Script ([src/train.py](src/train.py))
- Comet ML integration
- Mixed precision training
- EarlyStopping
- Model checkpointing

### FastAPI App ([deployment/app.py](deployment/app.py))
- Image upload endpoint
- Disease prediction
- Recommendation generation
- Health checks

---

## 📞 Support

- **Issues**: Check [SETUP_VERIFICATION.md](SETUP_VERIFICATION.md) troubleshooting section
- **Contribution**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Deployment**: Read [DEPLOYMENT.md](DEPLOYMENT.md)
- **Dataset**: Refer to [DATASET_ORGANIZATION.md](DATASET_ORGANIZATION.md)

---

## ✅ Status

**Project**: AgriAI-ViT  
**Version**: 1.0.0  
**Location**: `/Users/shreyaspatil/AgriAI-ViT/`  
**Status**: Files Written - Ready for Dataset Organization  
**Last Updated**: January 25, 2025

---

**Navigate to [TASK_COMPLETION_SUMMARY.md](TASK_COMPLETION_SUMMARY.md) for next steps.**
