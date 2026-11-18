# Robust Image Forgery Detection Under JPEG Compression

[![GitHub Repo](https://img.shields.io/badge/GitHub-Lehari1%2Fimage__forgery__detection-blue?logo=github)](https://github.com/Lehari1/image_forgery_detection)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg)](https://opencv.org/)
[![Task](https://img.shields.io/badge/Task-Image%20Tampering%20Detection-brightgreen.svg)]()
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-81.38%25-success.svg)]()

A deep learning pipeline designed to detect digital image tampering (splicing, copy-move, and retouching) with high robustness against post-processing JPEG compression artifacts. Built using PyTorch, the system leverages a **7-channel forensic representation** combining spatial color information, high-frequency noise residuals, and edge/gradient features.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Forensic Representation (7 Channels)](#-forensic-representation-7-channels)
- [Model Architecture](#-model-architecture)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Training Methodology](#-training-methodology)
- [Results](#-results)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Future Enhancements](#-future-enhancements)

---

## 🔍 Overview

Image manipulation and splicing often leave subtle digital traces such as inconsistencies in sensor noise, lighting, and edge transitions. However, standard forensic models fail when tampered images undergo lossy JPEG compression, which smooths out high-frequency forensic artifacts.

This project addresses this challenge by:
1. Combining original RGB images with **noise residuals** and **Laplacian edge responses** into a unified **7-channel input tensor**.
2. Training against multi-tier simulated JPEG compression factors ($Q \in \{30, 40, 50, 60, 70, 80, 90\}$).
3. Utilizing **Online Hard Example Mining (OHEM)** and **label smoothing** to focus learning on challenging tampering boundaries.

---

## ✨ Key Features

- **Multi-Modal Feature Fusion (7 Channels):** Integrates raw RGB pixels, Gaussian blur noise residuals, and normalized Laplacian edge maps.
- **Compression-Invariant Training:** Multi-quality offline dataset expansion combined with real-time online JPEG re-compression augmentation.
- **Deep CNN Classifier (`BetterCNN`):** Double-convolution blocks with Batch Normalization, ReLU activations, and Adaptive Average Pooling to prevent overfitting.
- **Robust Loss & Training Strategy:** Label-smoothed cross-entropy loss augmented with Online Hard Example Mining (top-70% hardest loss per batch) and gradient norm clipping.
- **Checkpoint Resilience:** Periodic batch-level and epoch-level checkpointing with automatic training resumption.

---

## 🔬 Forensic Representation (7 Channels)

Standard 3-channel (RGB) CNNs easily miss subtle manipulation artifacts hidden in frequency domain or sensor noise. This pipeline constructs a 7-channel tensor for every image:

```
Input Image (H x W x 3)
   ├── [Channels 0-2] Raw RGB: Preserves spatial context and semantic features.
   ├── [Channels 3-5] Noise Residual: Image minus Gaussian blur (5x5 kernel), extracting high-frequency sensor noise patterns.
   └── [Channel 6]    Laplacian Gradient: 2nd derivative edge map normalized to [0, 255], capturing tampering boundary discontinuities.
```

$$\text{Input Tensor} = \text{Concat}(\text{RGB}_{[3]}, \text{NoiseResidual}_{[3]}, \text{Laplacian}_{[1]}) \in \mathbb{R}^{7 \times 224 \times 224}$$

---

## 🏗 Model Architecture

The `BetterCNN` model is structured to process the 7-channel input through cascading feature extraction blocks:

```
Input (7 x 224 x 224)
  │
  ├── Block 1: Conv2d(7→32)  → BatchNorm → ReLU → Conv2d(32→32)  → BatchNorm → ReLU → MaxPool2d(2)  [32 x 112 x 112]
  ├── Block 2: Conv2d(32→64) → BatchNorm → ReLU → Conv2d(64→64)  → BatchNorm → ReLU → MaxPool2d(2)  [64 x 56 x 56]
  ├── Block 3: Conv2d(64→128)→ BatchNorm → ReLU → Conv2d(128→128)→ BatchNorm → ReLU → MaxPool2d(2)  [128 x 28 x 28]
  │
  ├── AdaptiveAvgPool2d((2, 2))  [128 x 2 x 2 = 512 dimensions]
  ├── Flatten
  ├── Linear(512 → 256) → ReLU → Dropout(p=0.3)
  └── Linear(256 → 2)   → Logits (Real / Tampered)
```

---

## 📊 Dataset & Preprocessing

The model is trained on the benchmark **CASIA 2.0 Image Tampering Detection Dataset**:
- **Authentic (`Au`):** Unaltered natural photographs.
- **Tampered (`Tp`):** Spliced and copy-move manipulated images.

### Data Generation Pipeline:
1. **Multi-Quality Compression Generation:** Images are exported across JPEG quality factors $Q \in \{30, 40, 50, 60, 70, 80, 90\}$.
2. **Dataset Composition:** Subsets from multiple quality tiers ($Q30, Q50, Q70, Q90$) are concatenated into a consolidated training pool.
3. **Data Augmentations:**
   - 50% chance of random dynamic JPEG re-compression.
   - 50% random horizontal flip.
   - 50% random vertical flip.
   - 30% random 90-degree orthogonal rotation.
   - Resized to uniform $224 \times 224$ resolution and normalized to $[0, 1]$.

---

## ⚙️ Training Methodology

| Parameter | Configuration |
|---|---|
| **Optimizer** | Adam ($\text{lr} = 5\times 10^{-5}$, $\text{weight\_decay} = 1\times 10^{-4}$) |
| **LR Scheduler** | StepLR ($\text{step\_size} = 10$, $\gamma = 0.5$) |
| **Loss Function** | Cross-Entropy with Label Smoothing ($0.1$) + Hard Example Mining (OHEM) |
| **Hard Mining Ratio** | Top 70% hardest loss samples per mini-batch |
| **Gradient Clipping** | Max norm $= 1.0$ |
| **Batch Size** | 16 |
| **Train/Test Split** | 80% Train / 20% Test |
| **Epochs** | 35 |

---

## 📈 Results

On an independent test set composed of mixed-quality compressed images:

| Metric | Score |
|---|---|
| **Test Accuracy** | **~81.38%** |
| **Resilience** | Robust against heavy JPEG compression ($Q=30$ through $Q=90$) |

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Lehari1/image_forgery_detection.git
cd image_forgery_detection
```

### 2. Install Dependencies
```bash
pip install torch torchvision numpy opencv-python pillow kaggle
```

*(GPU with CUDA support is recommended for training and evaluation).*

### 3. Download Dataset (Kaggle)
Set up your Kaggle API token (`kaggle.json` or environment variables):
```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_key"

kaggle datasets download -d divg07/casia-20-image-tampering-detection-dataset
unzip casia-20-image-tampering-detection-dataset.zip
```

---

## 💻 Usage

### Running in Google Colab / Jupyter Notebook
Open and run `Compressed_images_forgery (6).ipynb` directly on Google Colab with GPU runtime enabled:
1. Run the dataset setup cells to fetch and decompress CASIA 2.0.
2. Execute the synthetic compression generation script to create `Q30`, `Q50`, `Q70`, and `Q90` subsets.
3. Train the model using the training loop cell. Checkpoints will be saved as `last_checkpoint.pth` and `final_model.pth`.
4. Run the evaluation cell to assess classification accuracy on the test set.

---

## 📂 Project Structure

```text
├── Compressed_images_forgery (6).ipynb   # End-to-end Jupyter/Colab notebook
├── README.md                             # Project documentation
├── CASIA2/                               # CASIA 2.0 raw dataset (unzipped)
│   ├── Au/                               # Authentic images
│   └── Tp/                               # Tampered images
└── data/                                 # Generated compressed datasets
    ├── original/
    ├── Q30/
    ├── Q50/
    ├── Q70/
    └── Q90/
```

---

## 🔮 Future Enhancements

- [ ] Incorporate Error Level Analysis (ELA) and SRM (Spatial Rich Model) filter banks.
- [ ] Add dual-branch architecture separating frequency-domain (DCT coefficients) and spatial features.
- [ ] Implement localization / heatmap segmentation masks to highlight forged regions.
- [ ] Test generalization on benchmark datasets (e.g., COVERAGE, NIST16, Columbia).

---

## 👤 Author
- **Lehari1** - [GitHub Profile](https://github.com/Lehari1)
- Repository: [image_forgery_detection](https://github.com/Lehari1/image_forgery_detection)

---

## 📜 License & Acknowledgments
- Dataset: [CASIA 2.0 Tampering Dataset](https://www.kaggle.com/datasets/divg07/casia-20-image-tampering-detection-dataset) by CASIA (Institute of Automation, Chinese Academy of Sciences).
- Built with [PyTorch](https://pytorch.org/) and [OpenCV](https://opencv.org/).
