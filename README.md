# Recurrent Neural Networks for IMDB Sentiment Analysis

This project implements and evaluates multiple deep learning architectures for sentiment classification using the **IMDB Movie Reviews Dataset**. It spans from raw text cleaning and custom sequence processing to mathematical parameter verification, recurrent gradient analysis, hidden unit sensitivity studies, and end-to-end model benchmarking.

#Video Link :- https://drive.google.com/file/d/1NcYgS4GYt9FhCP36taaz32H3l0qNG6fY/view?usp=sharing
---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Implemented Architectures](#%EF%B8%8F-implemented-architectures)
3. [Key Experiments & Insights](#-key-experiments--insights)
4. [Performance Benchmark](#-performance-benchmark)
5. [File Deliverables](#-file-conclusions--deliverables)
6. [How to Reproduce](#%EF%B8%8F-how-to-reproduce)

---

## 🔍 Project Overview

The goal of this project is to perform binary sentiment analysis (Positive vs. Negative reviews) and explore the theoretical and practical limits of Feedforward vs. Recurrent Neural Networks.

### Core Pipeline Steps:
- **Text Preprocessing:** Strip HTML tags, remove non-alphabetic characters, lowercasing, and duplicate removal (retaining evaluation integrity).
- **Vocabulary Setup:** Tokenization on training data using the top 10,000 words. Sequences are padded to the 95th percentile of word counts (`MAXLEN = 587`) using `pre-padding` and `pre-truncation`.
- **Low-Level RNN Implementations:** Raw NumPy simulation of RNN hidden states and LSTM gates to verify alignment with high-level TensorFlow layers.
- **Gradient Analysis:** Logging gradient norms across time horizons to study the vanishing gradient problem and the impact of gradient clipping.
- **Sensitivity Analysis:** Evaluating parameter count scale vs. validation performance across GRUs and LSTMs with varying hidden unit dimensions.

---

## 🛠️ Implemented Architectures

1. **ANN Baseline (Feedforward):** An embedding-to-dense feedforward baseline that uses a `Flatten` layer before classification.
2. **SimpleRNN:** A single-layer standard Recurrent Neural Network designed to test vanilla BPTT dynamics.
3. **LSTM (Long Short-Term Memory):** Introduces memory cells and gating systems (Forget, Input, Candidate, Output) to mitigate gradient degradation.
4. **GRU (Gated Recurrent Unit):** A streamlined version of the LSTM utilizing a combined Update/Reset gate structure for computational efficiency.
5. **Stacked LSTM:** A deep recurrent network combining two sequentially stacked LSTM layers to learn higher-level semantic hierarchies.

---

## 📈 Key Experiments & Insights

* **Gradient Backpropagation Through Time (BPTT):** Vanilla SimpleRNN experiences dramatic gradient decay over long sequences, with gradient norms falling below computational precision over early steps. Both LSTM and GRU maintain healthy gradient flows across 580+ time steps.
* **Sequence Length Sensitivity:** Shorter sequence horizons (e.g., `MAXLEN = 50`) result in lower classifier accuracy due to information loss. However, extending sequence limits past optimal boundaries introduces redundant padding which harms model convergence.
* **Hidden State Dimensionality:** Increasing hidden states from `32` to `128` increases performance up to a threshold, beyond which overfitting occurs. GRUs consistently achieve comparable accuracy to LSTMs with roughly $25\% $ fewer parameters.

---

## 📊 Performance Benchmark

All models were evaluated on the held-out test split of 9,917 reviews:

| Model | Parameters | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ANN Baseline** | ~3.5M | ~84.3% | - | - | - | - |
| **SimpleRNN** | 62.0K | ~81.7% | 0.8177 | 0.8176 | 0.8175 | 0.8957 |
| **Stacked LSTM** | 357.2K | ~88.3% | 0.8835 | 0.8834 | 0.8834 | 0.9498 |
| **GRU (Best Model)** | 338.8K | **89.7%** | **0.8975** | **0.8973** | **0.8973** | **0.9576** |

---

## 💾 File Conclusions & Deliverables

The following evaluation assets are dynamically saved to `/content/plots/`:
- `model_comparison.csv` and `model_comparison.png`: Tabulated performance metrics.
- `roc_curves.png`: Combined ROC visual curves comparing model discriminative power.
- `sequence_robustness.png`: Accuracy metrics sliced by review lengths (Short, Medium, Long).
- `misclassified_reviews.csv`: Logged false positives and negatives for diagnostic profiling.
- `custom_inference_results.csv`: Predicted sentiment scores on complex user-defined sentences.
- `requirements.txt`: Package lock-file for quick dependency setup.

---

## ⚙️ How to Reproduce

To install dependencies and re-run all model training runs, run the following commands:

```bash
# Install project dependencies
pip install -r requirements.txt

# Run the Python notebook pipeline cells sequentially to rebuild metrics
```
