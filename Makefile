.PHONY: help install setup data train evaluate deploy test clean

# Default target
help:
	@echo "AgriAI-ViT Makefile Commands:"
	@echo "  make install       - Install all dependencies"
	@echo "  make setup         - Setup project (install + create directories)"
	@echo "  make data          - Download and prepare datasets"
	@echo "  make train         - Train the ViT model"
	@echo "  make evaluate      - Evaluate trained model"
	@echo "  make deploy        - Deploy FastAPI application"
	@echo "  make test          - Run all tests"
	@echo "  make clean         - Clean temporary files"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"

# Installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Setup project
setup: install
	mkdir -p data/raw data/processed models logs notebooks
	cp .env.example .env
	@echo "Setup complete! Please edit .env with your credentials."

# Data preparation
data:
	python src/data_preparation.py

# Training
train:
	python src/train.py

# Evaluation
evaluate:
	python src/evaluate.py

# Deploy API locally
deploy:
	uvicorn deployment.app:app --host 0.0.0.0 --port 8000 --reload

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html

# Clean temporary files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage

# Docker commands
docker-build:
	docker build -t agriai-vit:latest -f deployment/Dockerfile .

docker-run:
	docker run -p 8000:8000 --env-file .env agriai-vit:latest

docker-compose-up:
	docker-compose -f deployment/docker-compose.yml up -d

docker-compose-down:
	docker-compose -f deployment/docker-compose.yml down

# Notebook
notebook:
	jupyter notebook notebooks/

# Format code
format:
	black src/ tests/ deployment/
	isort src/ tests/ deployment/

# Lint code
lint:
	flake8 src/ tests/ deployment/
	pylint src/ tests/ deployment/

# Full pipeline
pipeline: setup data train evaluate

# Development mode
dev:
	uvicorn deployment.app:app --reload --log-level debug
