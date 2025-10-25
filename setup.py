"""
Setup script for AgriAI-ViT
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="agriai-vit",
    version="1.0.0",
    author="AgriAI Team",
    author_email="contact@agriai.com",
    description="Vision Transformer-Powered AI Recommendations for Smart Agriculture in India",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shreyaspatillll/AgriAI-ViT",
    project_urls={
        "Bug Tracker": "https://github.com/shreyaspatillll/AgriAI-ViT/issues",
        "Documentation": "https://github.com/shreyaspatillll/AgriAI-ViT/wiki",
        "Source Code": "https://github.com/shreyaspatillll/AgriAI-ViT",
    },
    packages=find_packages(exclude=["tests", "notebooks", "deployment"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "pylint>=2.15.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "agriai-train=src.train:main",
            "agriai-evaluate=src.evaluate:main",
            "agriai-predict=src.inference:main",
            "agriai-api=deployment.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "agriculture",
        "computer vision",
        "vision transformer",
        "deep learning",
        "crop disease detection",
        "smart farming",
        "AI",
        "machine learning",
    ],
)
