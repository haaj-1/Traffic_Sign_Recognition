# German Traffic Sign Recognition: CNN vs Vision Transformer

A deep learning study comparing a custom Convolutional Neural Network (CNN) and a Vision Transformer (ViT) built from scratch, applied to the classification of 43 German traffic sign categories.

---

## Executive Summary

This project investigates whether a Vision Transformer can match or exceed a CNN on a real-world image classification task under constrained compute conditions. Using the [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.kaggle.com/datasets/harbhajansingh21/german-traffic-sign-dataset), both architectures were trained, evaluated, and hyperparameter-tuned using Keras Tuner. The baseline ViT (93.95%) actually matched the baseline CNN (93.46%), suggesting the ViT is a competitive architecture for this task. However, the tuned ViT (89.07%) fell short of the tuned CNN (96.61%) — not necessarily because the architecture is inferior, but because the hyperparameter search was cut to only 3 of the intended 10 trials due to a limited GPU quotas. The final ViT was also stopped early before full convergence. These compute constraints mean the ViT results represent a lower bound on what the architecture could achieve. A complete tuning run could very likely close the gap with the CNN significantly, seeing as the base model slightly outperformed the CNN model.

---

## Model Selection Rationale

**Why CNN?**
CNNs have historically proven efficient and accurate for image classification tasks, making them a strong baseline with well-established benchmarks to compare against. Their inductive biases, local connectivity and translation invariance, align naturally with traffic sign images, where signs can appear at various positions and angles within a scene. CNNs are inherently good at extracting fine-grained local features like specific symbols, numbers, and edges, which are the defining characteristics that distinguish one traffic sign from another. They are also computationally efficient, making them well-suited for real-time applications such as those required in autonomous vehicles and intelligent transportation systems.

**Why ViT?**
The ViT was selected to explore the recent breakthrough that has challenged the long-standing dominance of CNNs in computer vision, and to investigate how viable an alternative it is for this specific task. The self-attention mechanism allows the ViT to capture broader scene context around a sign, which is something CNNs can struggle with due to their local receptive fields. Processing images as sequences of patch tokens also gives the ViT flexibility when dealing with signs at varied scales and distances. Beyond the immediate results, evaluating a standalone ViT contributes to the growing body of research identifying the conditions under which ViTs outperform CNNs, and could provide a foundation for exploring hybrid architectures in future work.

---

## Methodology

**1. Exploratory Data Analysis**
- Visualised all 43 sign classes and inspected sample images
- Analysed class distribution across train/validation/test splits, identifying significant imbalance (some classes have 10x more samples than others)
- Checked for corrupted or zero-value images

**2. Data Preprocessing & Augmentation**
- One-hot encoded class labels for categorical cross-entropy training
- Applied data augmentation to the training set (rotation ±10°, zoom ±10%, width/height shift ±10%) to address class imbalance and improve generalisation
- Normalised pixel values to [0, 1]

**3. CNN — Build, Train, Tune**
- Built a sequential CNN with Conv2D + BatchNorm + MaxPool + Dropout blocks
- Trained with Adam optimiser and early stopping
- Ran RandomSearch hyperparameter tuning (10 trials) over filter sizes, dense units, dropout rate, and learning rate

**4. Vision Transformer — Build, Train, Tune**
- Implemented a ViT from scratch: patch embedding → positional encoding → multi-head self-attention → MLP head
- Each 32×32 image is split into 64 patches of size 4×4
- Trained with early stopping; each epoch takes ~120s vs ~6s for the CNN
- Ran RandomSearch tuning (3 of 10 trials completed before GPU quota was exhausted)

**5. Evaluation**
- Accuracy and loss curves per epoch
- ROC-AUC (one-vs-rest, per class)
- Confusion matrix
- Per-class classification report (precision, recall, F1)

---

## Skills

- **Deep Learning** — CNN and Transformer architecture design from scratch in TensorFlow/Keras
- **Computer Vision** — image preprocessing, augmentation, multi-class classification
- **Hyperparameter Optimisation** — automated search with Keras Tuner (RandomSearch)
- **Model Evaluation** — ROC-AUC, confusion matrices, classification reports, training curve analysis
- **Data Analysis** — class imbalance detection, EDA, distribution visualisation with Matplotlib and Seaborn
- **Python** — NumPy, Pandas, scikit-learn, Pickle
- **MLOps basics** — early stopping, batch normalisation, dropout regularisation, callback management
- **Google Colab** — GPU-accelerated training, managing compute constraints

---

## Results

### CNN

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Baseline CNN | 93.46% | 0.3429 |
| Tuned CNN (best trial) | 96.01% | 0.1743 |
| **Final Tuned CNN** | **96.61%** | **0.1593** |

### Vision Transformer (ViT)

| Model | Test Accuracy | Test Loss |
|---|---|---|
| Baseline ViT | 93.95% | 0.2732 |
| Tuned ViT (3/10 trials) | 90.09% | 0.4939 |
| Final Tuned ViT | 89.07% | 0.5473 |

**Key finding:** At the baseline level, the ViT (93.95%) slightly outperformed the CNN (93.46%), suggesting it is a genuinely competitive architecture for this task. However, after tuning, the CNN (96.61%) pulled ahead of the ViT (89.07%). This reversal is largely attributed to the GPU quota constraint as the ViT tuning search only completed 3 of 10 trials. The ViT training curves also show the model had not converged by the time GPU time ran out — it was still improving at epoch 30 of the baseline run.

---

## Next Steps

Given more time and compute, the natural extensions to this project would be:

- **Complete the ViT tuning search** — run all 10 trials to find a properly regularised configuration; the 3-trial result is not representative
- **Transfer learning** — fine-tune a pre-trained ViT (e.g. `google/vit-base-patch16-224` from HuggingFace) on GTSRB; pre-trained ViTs consistently outperform CNNs when fine-tuned
- **Higher resolution inputs** — upscale images from 32×32 to 224×224; ViTs benefit significantly from larger patch counts and finer spatial detail
- **Class-weighted loss** — address the imbalance more directly by penalising misclassification of minority classes more heavily during training
- **Ensemble** — combine CNN and ViT predictions; the two architectures make different types of errors so an ensemble would likely outperform either individually
- **Deployment** — wrap the best model in a simple inference API (FastAPI or Flask) and build a demo that classifies traffic sign images in real time

---

## How to Run

1. Open `traffic_sign_recognition.ipynb` in Google Colab
2. Run the first cell to install `opendatasets`
3. Provide your Kaggle API credentials when prompted to download the dataset
4. Run all cells sequentially

> Developed and executed on Google Colab with GPU acceleration. The ViT sections are compute-intensive — a GPU runtime is required.

---

## Tech Stack

Python 3.11 · TensorFlow/Keras · Keras Tuner · NumPy · Pandas · Matplotlib · Seaborn · scikit-learn · Google Colab
