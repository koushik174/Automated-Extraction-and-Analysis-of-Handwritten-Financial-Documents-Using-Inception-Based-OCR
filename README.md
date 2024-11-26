# Handwritten Financial Document OCR System

## Overview
This project implements an advanced Optical Character Recognition (OCR) system specifically designed for handwritten financial documents. It uses the Inception V3 architecture with transfer learning to achieve high accuracy in character recognition, supporting both alphabets (A-Z) and numbers (0-9).

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## Features
- 🖼️ Advanced image preprocessing pipeline
- 🔍 Robust character recognition (A-Z, 0-9)
- 🎯 High accuracy with Inception V3 architecture
- 📊 Comprehensive evaluation metrics
- 🚀 Easy-to-use inference API
- 📈 Support for financial document analysis
- 🔄 Data augmentation capabilities
- 📝 Detailed logging and model versioning

## Project Structure
```
handwritten_ocr/
│
├── dataset/                      # Data directory
│   ├── raw_images/              # Original images
│   │   ├── train/
│   │   └── validation/
│   └── preprocessed_images/      # Processed images
│       ├── train/
│       └── validation/
│
├── models/                       # Model directory
│   ├── checkpoints/             # Model checkpoints
│   └── handwritten_ocr_model.h5 # Trained model
│
├── notebooks/                    # Jupyter notebooks
│   ├── EDA.ipynb                # Exploratory Data Analysis
│   ├── Data_Augmentation.ipynb  # Data augmentation examples
│   └── Run_Demo.ipynb           # Demo notebook
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── config.py                # Configuration settings
│   ├── data_loading.py          # Data loading utilities
│   ├── data_preprocessing.py    # Preprocessing pipeline
│   ├── model_training.py        # Model training code
│   ├── model_evaluation.py      # Evaluation metrics
│   ├── inference.py             # Inference utilities
│   └── utils.py                 # Helper functions
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_model.py
│
├── logs/                        # Training logs
├── requirements.txt             # Dependencies
├── setup.py                     # Package setup
├── LICENSE                      # License file
└── README.md                    # This file
```

## Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- At least 8GB RAM
- 20GB disk space

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/handwritten_ocr.git
cd handwritten_ocr
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up the dataset:
```bash
python src/data_loading.py
```

## Usage

### Training the Model

1. Preprocess the data:
```bash
python src/data_preprocessing.py
```

2. Train the model:
```bash
python src/model_training.py
```

3. Evaluate the model:
```bash
python src/model_evaluation.py
```

### Using Pre-trained Model

```python
from src.inference import predict_characters
from src.utils import preprocess_image_for_inference
import cv2

# Load image
image = cv2.imread('path_to_your_image.jpg')

# Preprocess image
processed_image = preprocess_image_for_inference(image)

# Get prediction
predicted_text = predict_characters(processed_image)
print(f"Predicted Text: {predicted_text}")
```

## Model Architecture
The system uses a modified Inception V3 architecture with the following enhancements:
- Transfer learning from ImageNet weights
- Custom top layers for character recognition
- Dropout layers for regularization
- Global average pooling
- Softmax activation for classification

## Training

### Dataset
The model is trained on the Bentham dataset, which includes:
- Handwritten characters (A-Z, 0-9)
- Various writing styles
- Different pen types and thicknesses

### Training Process
1. Data preprocessing
   - Image normalization
   - Noise removal
   - Orientation correction
   - Binarization
   - Data augmentation

2. Model training
   - Transfer learning
   - Fine-tuning
   - Early stopping
   - Learning rate scheduling

## Evaluation
The model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Character Error Rate (CER)

## API Reference

### Preprocessing
```python
from src.data_preprocessing import preprocess_image

# Preprocess single image
processed_image = preprocess_image(image_path)
```

### Training
```python
from src.model_training import train_model

# Train the model
model = train_model()
```

### Inference
```python
from src.inference import predict_characters

# Predict characters
prediction = predict_characters(model, image)
```

