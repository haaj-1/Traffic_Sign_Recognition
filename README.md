# German Traffic Sign Recognition — CNN vs Vision Transformer

A deep learning project comparing Convolutional Neural Networks (CNN) and Vision Transformers (ViT) for classifying German traffic signs, using the [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.kaggle.com/datasets/harbhajansingh21/german-traffic-sign-dataset) dataset.

---

## Project Overview

This project explores image classification on 43 traffic sign categories using two architectures:

- A custom **CNN** with batch normalization, dropout, and data augmentation
- A custom **Vision Transformer (ViT)** built from scratch with multi-head self-attention

Both models were trained, evaluated, and hyperparameter-tuned using **Keras Tuner (RandomSearch)** to find optimal configurations.

---

## Dataset

| Split      | Samples |
|------------|---------|
| Training   | 34,799  |
| Validation | 4,410   |
| Test       | 12,630  |

- Image size: 32×32×3 (RGB)
- Classes: 43 traffic sign categories
- Source: GTSRB via Kaggle

---

## Pipeline

1. **EDA** — class distribution analysis, sample visualization, class imbalance inspection
2. **Preprocessing** — grayscale conversion, normalization (÷255), one-hot encoding
3. **Augmentation** — rotation, zoom, width/height shift via `ImageDataGenerator`
4. **Model 1: CNN** — Conv2D + BatchNorm + MaxPool + Dropout + Dense layers
5. **Model 2: ViT** — patch embedding, positional encoding, multi-head attention, MLP head
6. **Hyperparameter Tuning** — Keras Tuner RandomSearch on both architectures
7. **Evaluation** — accuracy, loss curves, ROC-AUC, confusion matrix, classification report

---

## Results

### CNN

| Model                  | Test Accuracy | Test Loss |
|------------------------|---------------|-----------|
| Baseline CNN           | 93.46%        | 0.3429    |
| Tuned CNN (best trial) | 96.01%        | 0.1743    |
| Final Tuned CNN        | **96.61%**    | 0.1593    |

### Vision Transformer (ViT)

| Model                  | Test Accuracy | Test Loss |
|------------------------|---------------|-----------|
| Baseline ViT           | 93.95%        | 0.2732    |
| Tuned ViT (best trial) | 90.09%        | 0.4939    |
| Final Tuned ViT        | 89.07%        | 0.5473    |

**Key finding:** The CNN outperformed the ViT on this dataset. ViTs typically require larger datasets and more compute to reach their full potential — at 32×32 resolution with ~35k training samples, the CNN's inductive biases (local feature extraction, translation invariance) gave it a clear edge.

---

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- Keras Tuner
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- Google Colab (GPU runtime)

---

## How to Run

1. Open `traffic_sign_recognition.ipynb` in Google Colab
2. Run the first cell to install `opendatasets`
3. Provide your Kaggle API credentials when prompted to download the dataset
4. Run all cells sequentially

> The notebook was developed and executed on Google Colab with GPU acceleration.

---

## Key Concepts 

- Custom CNN architecture design with regularization techniques
- Vision Transformer implementation from scratch (patch embedding, positional encoding, multi-head self-attention)
- Data augmentation strategies for imbalanced image datasets
- Automated hyperparameter search with Keras Tuner
- Model evaluation with ROC-AUC curves, confusion matrices, and classification reports
- Comparative analysis of CNN vs Transformer architectures on small-scale vision tasks
