<div align="center">

# 🥬 Vegetable Classification with DINOv2

### A lightweight image-classification pipeline powered by pretrained vision transformers and reusable image embeddings.

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-DINOv2-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="DINOv2">
  <img src="https://img.shields.io/badge/Task-Image%20Classification-8A2BE2?style=for-the-badge" alt="Image classification">
</p>

**Classify 15 types of vegetables using 768-dimensional visual embeddings extracted from a frozen DINOv2 backbone.**

</div>

---

## 📌 Overview

This project explores an efficient approach to image classification using **[DINOv2-Base](https://huggingface.co/facebook/dinov2-base)**. Instead of fine-tuning the entire vision transformer, the pretrained backbone is kept frozen and used as a feature extractor.

The image embeddings are extracted once, saved, and reused to train a small neural-network classifier. This separates the expensive image-feature extraction stage from the lightweight classifier-training stage.

The project uses the **[Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)**, containing 21,000 images across 15 vegetable categories.

## ✨ Highlights

- 🧠 **Pretrained DINOv2 backbone** — uses `facebook/dinov2-base` from Hugging Face.
- 🔒 **Frozen feature extractor** — keeps the pretrained backbone weights fixed.
- ⚡ **Reusable embeddings** — extracts and saves 768-dimensional image representations so they do not need to be recomputed for every training run.
- 🎯 **Lightweight classifier** — Dropout followed by a Linear layer for 15-class prediction.
- 📊 **Detailed evaluation** — accuracy, loss, confusion matrix, and per-class precision, recall, and F1-score.
- 🧪 **Separate data splits** — train, validation, test, and combined train-validation datasets.
- 🛠️ **Reusable utilities** — uses a custom `pytorch_utils` module for dataset creation, DataLoaders, training, evaluation, and plotting.

## 📈 Results

The notebook reports the following results:

| Evaluation | Accuracy | Macro F1-score |
|---|---:|---:|
| Validation experiment | **99.97%** | **99.97%** |
| Final test evaluation | **99.93%** | **99.93%** |

The validation classification report gives a macro F1-score of approximately **99.9667%**. The final test classification report gives approximately **99.9333%**.

> **Evaluation note:** The final train-plus-validation run uses the test DataLoader during the training helper's best-epoch selection. This means the final test result is not a fully untouched estimate and may be slightly optimistic. The validation experiment is the cleaner generalization measurement. For a strictly independent final test, select the epoch using validation data first, then evaluate the test set only once.

## 🥕 Dataset

**Source:** [Vegetable Image Dataset — Kaggle](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)

The dataset is organized into class-specific folders and contains:

| Split | Images |
|---|---:|
| Training | 15,000 |
| Validation | 3,000 |
| Test | 3,000 |
| **Total** | **21,000** |

There are **15 classes**:

`Bean`, `Bitter_Gourd`, `Bottle_Gourd`, `Brinjal`, `Broccoli`, `Cabbage`, `Capsicum`, `Carrot`, `Cauliflower`, `Cucumber`, `Papaya`, `Potato`, `Pumpkin`, `Radish`, and `Tomato`.

The notebook uses `torchvision.datasets.ImageFolder` through the custom `create_datasets()` utility.

## 🏗️ Approach

The workflow has two main stages.

### 1. Extract image embeddings

1. Load the pretrained `facebook/dinov2-base` model.
2. Freeze the backbone parameters.
3. Resize images to **224 × 224** and normalize them using ImageNet mean and standard deviation.
4. Pass each image through DINOv2.
5. Take `pooler_output` (the image-level CLS representation) as the image embedding.
6. Save embeddings and labels for each data split in a `.pt` file.

Each image is represented by a **768-dimensional vector**.

### 2. Train the classifier

The extracted embeddings are wrapped in `TensorDataset` objects and loaded through DataLoaders. A small classifier is trained on these vectors.

```text
Input image (224 × 224)
        │
        ▼
Frozen DINOv2-Base
        │
        ▼
Image-level CLS embedding
        │
        ▼
768-dimensional feature vector
        │
        ▼
Dropout (p = 0.3)
        │
        ▼
Linear layer (768 → 15)
        │
        ▼
Predicted vegetable class
```

### Classifier

```python
class Classifier(nn.Module):
    def __init__(self, input_features: int, num_classes: int, dropout: float):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_features, num_classes)
        )

    def forward(self, x):
        return self.fc(x)
```

### Training configuration

| Setting | Value |
|---|---|
| Backbone | `facebook/dinov2-base` |
| Embedding size | 768 |
| Input image size | 224 × 224 |
| Number of classes | 15 |
| Classifier | Dropout + Linear |
| Dropout | 0.3 |
| Loss function | Cross Entropy Loss |
| Optimizer | AdamW |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Epochs | 20 |

No random augmentation is applied in the embedding-extraction pipeline. Since embeddings are computed once and reused, random augmentation at extraction time would be captured in the saved features rather than varied across classifier-training epochs.

## 🧰 Tech Stack

- **Python**
- **PyTorch** and **Torchvision**
- **Hugging Face Transformers**
- **DINOv2-Base**
- **scikit-learn** — confusion matrix
- **PyTorch Ignite** — classification report
- **Matplotlib** — plots and evaluation visualizations
- **Kaggle Notebooks** — dataset and GPU environment

## 📂 Project Structure

A suggested repository layout:

```text
vegetable-classification/
│
├── vegetable-classification.ipynb
├── pytorch_utils.py
├── dinov2_custom_classifier.pt
├── dinov2_vegetables_embeddings.pt   # Optional: cached embeddings
│
└── README.md
```

The notebook saves the trained classifier weights as `dinov2_custom_classifier.pt`. The embedding cache is saved separately as `dinov2_vegetables_embeddings.pt` and can be reused to avoid extracting the image features again.

> The `.pt` files are generated artifacts; they are not necessarily included in this repository. Add them if you intend to distribute the trained weights or cached embeddings.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/vegetable-classification.git
cd vegetable-classification
```

### 2. Install dependencies

```bash
pip install torch torchvision transformers torchinfo matplotlib scikit-learn pytorch-ignite tqdm pillow
```

Also make sure the custom `pytorch_utils.py` module used by the notebook is available in the project directory or Python path.

### 3. Download the dataset

Download and extract the [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset).

Update the dataset path in the notebook to match your local environment:

```python
from pathlib import Path

PATH = Path("/path/to/Vegetable Images")
```

The expected folder structure is:

```text
Vegetable Images/
├── train/
│   ├── Bean/
│   ├── Bitter_Gourd/
│   └── ...
├── val/
│   ├── Bean/
│   ├── Bitter_Gourd/
│   └── ...
└── test/
    ├── Bean/
    ├── Bitter_Gourd/
    └── ...
```

### 4. Run the notebook

Open `vegetable-classification.ipynb` in Jupyter or Kaggle Notebooks and execute the cells in order.

The first run extracts embeddings and saves them. On subsequent runs, the notebook attempts to load the saved embedding file if it is available at the configured path.

## 💾 Saved Model

The notebook saves the classifier's best-epoch state dictionary:

```python
torch.save(best_model_state, "dinov2_custom_classifier.pt")
```

**Important:** this file contains only the classifier weights, not the DINOv2 backbone. To make predictions, load the same pretrained DINOv2 backbone, recreate the classifier architecture with 768 input features and 15 output classes, and load the classifier state dictionary.

The inference pipeline must use the same image preprocessing and class-index mapping as training.

## 🔍 Evaluation

The notebook evaluates the model using:

- **Accuracy and loss** to measure overall performance.
- **Confusion matrix** to inspect correct predictions and class-level errors.
- **Classification report** to inspect precision, recall, and F1-score for each vegetable.
- **Macro average** to give every class equal weight, regardless of its number of samples.

## 📚 References

- [DINOv2 — Hugging Face Model](https://huggingface.co/facebook/dinov2-base)
- [DINOv2 Paper — Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)
- [Vegetable Image Dataset — Kaggle](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision ImageFolder](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)

---

<div align="center">

**Built with PyTorch, DINOv2, and a lot of experimentation.** 🥬

</div>
