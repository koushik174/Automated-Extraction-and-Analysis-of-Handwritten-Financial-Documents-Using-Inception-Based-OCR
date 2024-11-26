# Automated-Extraction-and-Analysis-of-Handwritten-Financial-Documents-Using-Inception-Based-OCR
## Project Overview

This project aims to develop an OCR system tailored for handwritten financial documents. By fine-tuning the Inception v3 model, the system accurately extracts textual information from images of handwritten invoices, receipts, and other financial records. The extracted data is then converted into structured formats suitable for financial analysis.

## Features

- **Image Preprocessing**: Enhances image quality for better OCR accuracy through noise reduction, binarization, and skew correction.
- **Data Augmentation**: Increases dataset diversity using techniques like rotation, scaling, and elastic distortions.
- **Model Training**: Fine-tunes the Inception v3 model using transfer learning and custom layers for character recognition.
- **Model Evaluation**: Provides detailed evaluation metrics, including accuracy, precision, recall, and confusion matrices.
- **Inference Pipeline**: Offers a user-friendly script to perform OCR on new images and output structured data.
- **Utilities**: Contains helper functions for tasks like label encoding, character mapping, and visualization.
- **Testing Suite**: Includes unit tests to ensure the reliability of preprocessing functions and model predictions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dataset Preparation](#dataset-preparation)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.7 or higher
- Virtual environment (optional but recommended)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/handwritten-financial-ocr.git
   cd handwritten-financial-ocr
