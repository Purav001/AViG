# Attention-based Vision Graph Neural Network (AViG)

This repository contains the official PyTorch implementation for the project **"Attention-based Vision Graph Neural Network framework for general-purpose visual analysis"**.

Our work introduces the **Pyramid Vision Graph Network (Pyramid ViG)**, a novel architecture that represents images as flexible graphs to overcome the limitations of traditional grid-based CNNs and sequence-based Vision Transformers. The core contribution is a novel **Spatial Refinement Head**, which integrates a lightweight attention mechanism to intelligently focus on the most discriminative features, significantly enhancing performance across diverse visual tasks.

This framework is rigorously validated on three distinct and challenging domains, consistently outperforming established baselines.

---

## 🚀 Key Features

- **Graph-Based Vision:** Moves beyond rigid grids by representing images as graphs of interconnected patches, enabling flexible modeling of complex and irregular objects.
- **Novel Attention Head:** Implements a **Spatial Refinement Head** with a Squeeze-and-Excitation (SE) block that adaptively recalibrates channel-wise features for superior discriminative power.
- **State-of-the-Art Performance:** Achieves superior or highly competitive results on three diverse benchmarks:
  - **General Classification (CIFAR-10):** 84.9% Accuracy
  - **Medical Imaging (ISIC 2019):** 77.11% Accuracy
  - **Video Anomaly Detection (UCSD):** 0.91 AUROC
- **Versatile Backbone:** The Pyramid ViG backbone demonstrates strong generalization, proving its effectiveness as a powerful, all-purpose vision model.

---

## 🏛️ Architecture Overview

The model is composed of two main components: the **Pyramid ViG Backbone** and the **Spatial Refinement Head**.

### 1. Pyramid ViG Backbone

Inspired by the pioneering "Vision GNN" paper, our backbone processes images by:
1.  **Image to Graph:** Dividing an image into patches, which become nodes in a graph.
2.  **Dynamic Edges:** Connecting nodes based on feature similarity (K-Nearest Neighbors, K=9) rather than spatial proximity.
3.  **Hierarchical Processing:** Using a pyramid structure with four stages to learn multi-scale features, which is highly effective for visual tasks.
4.  **ViG Blocks:** Each stage consists of repeating **ViG Blocks**, which contain a **Grapher** module for aggregating neighborhood information and a **Feed-Forward Network (FFN)** to prevent over-smoothing and enhance feature diversity.

### 2. The Spatial Refinement Head (Our Contribution)

This is the key innovation of our project. Instead of a simple linear classifier, this head performs a final stage of intelligent feature refinement:

1.  **Channel Expansion:** A 1x1 convolution expands the feature map to create a richer representation.
2.  **Local Refinement:** A 3x3 convolution integrates local neighborhood context between the high-level feature vectors.
3.  **Attention (SE Block):** A Squeeze-and-Excitation block adaptively re-weights each channel's importance, allowing the model to focus on the most informative features and suppress noise.
4.  **Classification:** A final pooling and linear layer produce the classification output.

---

## 📊 Results

Our unified framework was benchmarked against strong baselines (ResNet-50 and ViT-S) and demonstrated superior performance across all tasks.

| Model             | CIFAR-10 Accuracy | ISIC 2019 Accuracy | UCSD AUC |
| :---------------- | :---------------: | :----------------: | :------: |
| ResNet-50         |        82%        |        72%         |   0.85   |
| ViT-S             |        84%        |        75%         |   0.88   |
| **ViG-S (Ours)** |     **84.90%** |     **77.11%** | **0.91** |

---

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Purav001/AViG.git](https://github.com/Purav001/AViG.git)
    cd AViG
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ How to Run

### 1. Prepare Datasets

- **CIFAR-10:** The `prepare_cifar10.py` script will automatically download and structure the dataset.
  ```bash
  python prepare_cifar10.py
