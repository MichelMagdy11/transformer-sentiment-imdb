 Transformer from Scratch — IMDB Sentiment Analysis

This project implements a **Transformer-based sentiment analysis model built completely from scratch using TensorFlow/Keras**, without relying on prebuilt Transformer layers.

The model is trained on the **IMDB movie reviews dataset** to classify reviews as **positive or negative sentiment**.

---

## 🚀 Project Highlights

- Implements core Transformer components manually:
  - Positional Encoding
  - Multi-Head Self-Attention
  - Position-wise Feed-Forward Network
  - Layer Normalization
  - Residual Connections
  - Encoder Blocks
- End-to-end sentiment classification pipeline
- Built using `tf.data` for efficient training
- Includes evaluation metrics (Accuracy, AUC, Precision, Recall, F1-score)
- Confusion matrix + training visualization
- Inference on custom reviews

---

## 🧠 Model Architecture

The model consists of:

1. **Embedding Layer**
2. **Positional Encoding**
3. **Transformer Encoder Blocks (stacked)**
4. **Global Average Pooling**
5. **Fully Connected Layers**
6. **Sigmoid Output Layer**

### Key Components Implemented From Scratch

- Multi-Head Self-Attention
- Scaled Dot-Product Attention
- Sinusoidal Positional Encoding
- Residual + LayerNorm blocks

---

## 📊 Dataset

- Dataset: IMDB Movie Reviews
- Source: Kaggle / TensorFlow Datasets version
- Classes:
  - Positive (1)
  - Negative (0)

Preprocessing steps:
- HTML tag removal
- Lowercasing
- Punctuation removal
- Tokenization using `TextVectorization`

---

## ⚙️ Hyperparameters

| Parameter      | Value  |
|----------------|--------|
| Vocabulary Size| 20,000 |
| Max Length     | 200    |
| Embedding Dim  | 64     |
| Attention Heads| 4      |
| FFN Dimension  | 128    |
| Layers         | 2      |
| Dropout        | 0.1    |
| Batch Size     | 64     |
| Epochs         | 5      |
| Learning Rate  | 1e-3   |

---

## 🏗️ Training

The model is trained using:

- Optimizer: Adam
- Loss: Binary Crossentropy
- Metrics:
  - Accuracy
  - AUC
  - Precision
  - Recall

Callbacks used:
- EarlyStopping (on validation accuracy)
- ReduceLROnPlateau

---

## 📈 Results

After training, the model is evaluated using:

- Test Accuracy
- AUC Score
- Precision / Recall
- F1 Score
- Confusion Matrix
