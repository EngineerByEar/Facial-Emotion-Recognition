# 😁 Facial Expression Recognition

This project explores facial expression recognition using the **FER2013** dataset. It benchmarks multiple model architectures from a lightweight custom CNN to various ResNet50 transfer learning configurations, to identify the most effective approach for classifying human emotions from grayscale face images.

## 🎯 Overview

The goal is to classify facial images into one of **7 emotion categories**:
`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

The project covers the full pipeline:
- Data loading and preprocessing
- Model design and training
- Hyperparameter and architecture experimentation
- Cross-model evaluation and comparison

---

## 🤖 Models

Seven model configurations were trained and evaluated:

| Model | Description |
|---|---|
| **SGD** | Linear classifier trained with stochastic gradient descent — simple baseline |
| **Custom CNN** | 3-layer convolutional network with batch normalization and dropout, trained from scratch |
| **ResNet50 — Single FC** | Pretrained ResNet50 with frozen backbone, single fully connected output layer |
| **ResNet50 — Complex FC** | Pretrained ResNet50 with a multi-layer classification head |
| **ResNet50 — Complex FC (Weighted)** | Same as above with class-weighted loss to address class imbalance |
| **ResNet50 — Complex FC Last Layer (Weighted)** | Only the final layer unfrozen, with class weighting |
| **ResNet50 — Full (Weighted)** | Full ResNet50 fine-tuning with class-weighted loss |

---

## 📊 Results

| Model | Accuracy |
|---|---|
| **Custom CNN** | **56.20%** ✅ Best |
| ResNet50 — Complex FC (Full) | 42.41% |
| ResNet50 — Single FC | 41.89% |
| ResNet50 — Complex FC Full (Weighted) | 41.59% |
| ResNet50 — Complex FC (Weighted) | 39.75% |
| ResNet50 — Complex FC Last Layer (Weighted) | 39.68% |
| SGD | 37.87% |

The **Custom CNN** outperformed all ResNet50 configurations, likely due to the domain mismatch between ImageNet pretraining and grayscale facial expression images in lower resolution.

---

## 🏗️ Architecture

### Custom CNN (`cnn_model.py`)

```
Input (1×48×48)
  → Conv2d(1→16) + BatchNorm + ReLU + MaxPool
  → Conv2d(16→32) + BatchNorm + ReLU + MaxPool
  → Conv2d(32→64) + BatchNorm + ReLU
  → Flatten → FC(9216→128) + Dropout(0.4)
  → FC(128→64) + Dropout(0.4)
  → FC(64→8)
```

### ResNet50 Variants

All ResNet50 models use pretrained ImageNet weights. Variants differ along two axes:
- **How much of the network is unfrozen** (last layer only vs. full fine-tuning)
- **Whether class-weighted loss** is applied to handle FER2013's class imbalance

---

## 📁 Data

**Dataset:** [FER2013](https://www.kaggle.com/datasets/msambare/fer2013)

FER2013 contains ~35,000 grayscale 48×48 pixel face images across 7 emotion classes. The dataset is notably imbalanced — `happy` is heavily overrepresented while `disgust` has very few samples.

Data loading is handled by `data_load_utils.py`, which reads images from a folder structure organized by emotion label and returns them as NumPy arrays alongside their labels.

---

## 🔬 Experimentation

The notebooks in this repository document the full experimentation process:

| Notebook | Description |
|---|---|
| `Training Custom CNN.ipynb` | Custom CNN training |
| `Training SGD CLF.ipynb` | SGD baseline |
| `Training ResNet50.ipynb` | ResNet50 single FC head |
| `Training ResNet50-complex-*.ipynb` | ResNet50 architecture variants |
| `Training ResNet50-Weight-Distributed.ipynb` | Distributed weight experiments |
| `Model Comparison.ipynb` | Cross-model evaluation |

---

## 🛠️ Tech Stack

- **Framework:** PyTorch
- **Pretrained Models:** torchvision (ResNet50, ResNet18)
- **Data Handling:** Pillow, NumPy, Pandas
- **Visualization:** Matplotlib

---

## 🚀 Possible Extensions

- Data augmentation (flipping, rotation, color jitter) to reduce overfitting
- Vision Transformer (ViT) as alternative backbone
- More complex CNN architectures
- Ensemble methods combining CNN and ResNet predictions
- Real-time inference via webcam
