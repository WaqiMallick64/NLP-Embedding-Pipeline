# Skip-Gram Word2Vec Embedding from Scratch using PyTorch

This project implements a simplified version of the **Skip-Gram Word2Vec model** using **PyTorch**, trained on a custom text corpus. It covers all key stages from preprocessing to embedding visualization.

---

## 🧠 Project Overview

**Objective**: Learn word embeddings using a Skip-Gram architecture and visualize semantic relationships between words.

---

## 🧱 Approach

### 🔹 Task 1: Preprocessing the Text
- Loaded and tokenized the text corpus using **NLTK**.
- Removed all punctuation using `string.punctuation`.
- Created two mappings:
  - `word2idx`: Word → Index
  - `idx2word`: Index → Word

### 🔹 Task 2: Skip-gram Model Implementation
- A context window of size 2 was used to generate `(target, context)` pairs.
- Built a PyTorch-based Skip-Gram model:
  - `nn.Embedding`: To learn dense vector representations.
  - `nn.Linear`: To project embeddings to vocabulary size for context prediction.
- Used `CrossEntropyLoss` as the loss function and **Adam optimizer** for training.

### 🔹 Task 3: Training and Saving Embeddings
- Model was trained for **100 epochs**.
- Observed steady loss reduction from ~140 to ~134.
- Final learned embeddings were stored in a Python dictionary `embeddings_dict`.
- (Optional) Embeddings can be saved to `.txt` for reuse.

### 🔹 Task 4: Embedding Visualization
- Applied **t-SNE** to reduce the 10D embeddings to 2D.
- Used **matplotlib** to visualize and annotate the embeddings.
- Some semantically related words such as `"data"`, `"AI"`, and `"learning"` appeared closer in the vector space (limited due to small corpus).

---

## 🧪 Dependencies

- Python 3.x
- PyTorch
- NLTK
- NumPy
- scikit-learn
- matplotlib

Install with:

```bash
pip install torch nltk numpy scikit-learn matplotlib
