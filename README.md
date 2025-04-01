# FreshHarvest: Fruit Freshness Detection with Deep Learning

![FreshHarvest Logo](https://via.placeholder.com/150) <!-- Replace with actual logo if available -->

**FreshHarvest** is an advanced machine learning project designed to classify the freshness of fruits using state-of-the-art computer vision techniques. Leveraging deep learning models and a robust dataset, this project provides an end-to-end solution for detecting whether a fruit is fresh or spoiled. It includes a modular codebase, a Flask-based web interface for real-time predictions, and comprehensive evaluation tools, making it a versatile tool for both research and practical deployment in agriculture or food quality assurance.

This project demonstrates expertise in deep learning, computer vision, data engineering, and software development, showcasing skills in model training, optimization, deployment, and user interface design.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Technical Architecture](#technical-architecture)
   - [Model Architecture](#model-architecture)
   - [Code Structure](#code-structure)
5. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Setup Instructions](#setup-instructions)
6. [Usage](#usage)
   - [Training the Model](#training-the-model)
   - [Evaluating the Model](#evaluating-the-model)
   - [Making Predictions](#making-predictions)
   - [Running the Web Interface](#running-the-web-interface)
7. [Scripts and Modules](#scripts-and-modules)
8. [Results and Performance](#results-and-performance)
9. [Future Improvements](#future-improvements)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgments](#acknowledgments)

## Project Overview

FreshHarvest aims to automate the process of fruit freshness detection, a critical task in agriculture, food supply chains, and quality control. By employing convolutional neural networks (CNNs) trained on a diverse dataset of fruit images, the system can distinguish between fresh and spoiled fruits with high accuracy. The project integrates several components:

- **Data Processing:** Custom dataset handling with augmentation and normalization.
- **Model Training:** Transfer learning with pre-trained CNN architectures (EfficientNet, ResNet, MobileNet).
- **Inference:** Real-time prediction capabilities for single images or batches.
- **Web Interface:** A user-friendly Flask application for uploading images and viewing results.
- **Evaluation:** Comprehensive metrics and visualizations (confusion matrix, accuracy, precision, recall, F1-score).

This project is built with modularity and scalability in mind, making it easy to extend to new fruit types or integrate into larger systems.

## Features

- **Multi-Architecture Support:** Choose from EfficientNet-B0, ResNet-50, or MobileNet-V2 for classification.
- **Transfer Learning:** Utilize pre-trained weights from ImageNet with options to freeze/unfreeze the backbone.
- **Data Augmentation:** Robust preprocessing with random flips, rotations, and color jittering.
- **Real-Time Inference:** Predict freshness on individual images or directories of images.
- **Web Application:** Upload images via a Flask-based interface and view annotated results.
- **Comprehensive Evaluation:** Generate detailed metrics and visualizations post-training.
- **Modular Design:** Clean, reusable code separated into distinct modules (data, model, training, etc.).
- **Early Stopping:** Prevent overfitting with configurable patience and minimum delta.
- **Visualization Tools:** Plot training history, confusion matrices, and prediction distributions.

## Dataset

The dataset used in FreshHarvest is the **FRUIT-16K** dataset (assumed structure based on code), stored at `C:/FreshHarvest Project/FreshHarvest_Dataset/FRUIT-16K`. It consists of images organized into folders prefixed with:
- `F_`: Fresh fruits
- `S_`: Spoiled fruits

Supported fruit types include:
- Banana
- Lemon
- Lulo
- Mango
- Orange
- Strawberry
- Tamarillo
- Tomato

The dataset is split into training (70%), validation (15%), and testing (15%) sets with stratification to maintain class balance. Images are resized to 224x224 pixels and normalized using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

## Technical Architecture

### Model Architecture

FreshHarvest leverages transfer learning with three optional CNN backbones:
1. **EfficientNet-B0:** Lightweight and efficient, ideal for resource-constrained environments.
2. **ResNet-50:** Deep residual network with strong generalization capabilities.
3. **MobileNet-V2:** Optimized for mobile and edge devices with inverted residuals.

Each model's final fully connected layer is replaced with a custom classifier:
- Dropout layer (configurable rate, default 0.2) for regularization.
- Linear layer mapping to 2 output classes (fresh, spoiled).

Training uses the Adam optimizer with a learning rate scheduler (`ReduceLROnPlateau`) and cross-entropy loss. Early stopping ensures optimal model selection.

### Code Structure

The project is organized into modular Python scripts:

- **`config.py`**: Centralized configuration (paths, hyperparameters, device settings).
- **`data.py`**: Dataset class, data loading, and augmentation utilities.
- **`model.py`**: Model definitions and utilities (architecture selection, parameter counting).
- **`trainer.py`**: Training, validation, and evaluation logic with early stopping.
- **`inference.py`**: Inference class for deployment and predictions.
- **`predict.py`**: Script for making predictions on new images or directories.
- **`evaluate.py`**: Script for evaluating the model on test data or custom images.
- **`main.py`**: Entry point with modes (train, evaluate, predict, serve).
- **`train.py`**: Dedicated training script with detailed logging and visualization.
- **`app.py`**: Flask web application for interactive predictions.
- **`utils.py`**: Helper functions (logging, plotting, seed setting).

## Installation

### Prerequisites

- Python 3.8+
- PyTorch (`torch`, `torchvision`)
- OpenCV (`opencv-python`)
- NumPy, Pandas, Matplotlib
- Flask
- Scikit-learn
- TQDM (progress bars)

See `requirements.txt` for the full list.

### Setup Instructions

1. **Clone the Repository:**


2. **Install Dependencies:**


3. **Set Up Dataset:**
- Place the `FRUIT-16K` dataset at `C:/FreshHarvest Project/FreshHarvest_Dataset/FRUIT-16K`.
- Ensure folder structure follows `F_<fruit_type>` and `S_<fruit_type>` conventions.

4. **Verify Configuration:**
- Update `config.py` if paths or parameters need customization.

## Usage

### Training the Model

Train a new model with:
- Outputs: Model checkpoint, training history, metrics, and visualizations in `model_outputs/run_<timestamp>`.

### Evaluating the Model

Evaluate a trained model:
- Access at `http://localhost:5000`.
- Upload images to get predictions with annotated results.

## Scripts and Modules

| Script/Module      | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `config.py`        | Configuration parameters (paths, hyperparameters, device).                 |
| `data.py`          | Dataset handling, transforms, and data loaders.                            |
| `model.py`         | CNN model definitions and utilities.                                       |
| `trainer.py`       | Training, validation, and evaluation functions with early stopping.        |
| `inference.py`     | Inference class for predictions and annotated image generation.            |
| `predict.py`       | Prediction script for single or batch inference with visualization options.|
| `evaluate.py`      | Evaluation script for test set or custom images with metrics.              |
| `main.py`          | Main entry point with mode selection (train, evaluate, predict, serve).    |
| `train.py`         | Dedicated training script with detailed logging and plotting.              |
| `app.py`           | Flask web app for interactive predictions.                                 |
| `utils.py`         | Utility functions (logging, plotting, seed setting).                       |

## Results and Performance

The model achieves strong performance on the test set (example metrics):
- **Accuracy:** ~92%
- **Precision:** ~91%
- **Recall:** ~93%
- **F1-Score:** ~92%

Visualizations include:
- Training history (loss/accuracy vs. epoch).
- Confusion matrix.
- Prediction distribution for batch inference.

Results vary based on architecture, dataset size, and training parameters. EfficientNet-B0 typically offers the best balance of accuracy and efficiency.

## Future Improvements

- **Expanded Dataset:** Include more fruit types and diverse conditions (lighting, angles).
- **Model Optimization:** Implement quantization and pruning for edge deployment.
- **Multi-Class Support:** Extend to classify fruit types alongside freshness.
- **Real-Time Processing:** Integrate with camera feeds for live detection.
- **Mobile App:** Develop a mobile interface using Flutter or React Native.
- **Hyperparameter Tuning:** Use tools like Optuna for automated tuning.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please follow the coding style and include tests where applicable.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- **PyTorch Team:** For an excellent deep learning framework.
- **OpenCV Community:** For robust computer vision tools.
- **Flask Developers:** For a lightweight web framework.
- **Dataset Contributors:** For providing the FRUIT-16K dataset (assumed).

*Created by Erick K Yegon, Machine Learning Specialist. Contact: keyegon@gmail.com*