# Operations on Word Vectors — Debiasing

> **Course:** Sequence Models — Week 2: Natural Language Processing & Word Embeddings  
> **Assignment:** 1

---

## Overview

This assignment explores **word embeddings** and how to work with them in practice. You'll load pre-trained GloVe vectors, measure semantic similarity using cosine similarity, and apply **debiasing algorithms** to reduce gender bias encoded in word representations.

---

## Project Structure

```
Assignment_1/
├── 22_Operations on Word Vectors - Debiasing.ipynb   # Main notebook
├── data/
│   ├── glove.6B.50d.txt                              # Pre-trained GloVe embeddings (see setup)
│   ├── input-Copy1.txt
│   └── input.txt
├── generateTestCases.py                              # Test case generator
├── images/                                           # Diagrams used in the notebook
│   ├── 1-hot-vector.png
│   ├── cosine_sim.png
│   ├── equalize10.png
│   ├── lookup.png
│   ├── neutralize_kiank.png
│   ├── neutralize.png
│   └── neutral.png
├── w2v_utils.py                                      # Helper utilities for word vectors
└── __pycache__/
```

---

## Setup

### 1. Download the GloVe embeddings

The notebook requires the **GloVe 6B 50-dimensional** word vectors.

1. Go to the Kaggle dataset page:  
   👉 [https://www.kaggle.com/datasets/watts2/glove6b50dtxt](https://www.kaggle.com/datasets/watts2/glove6b50dtxt)

2. Download `glove.6B.50d.txt` (you may need a free Kaggle account).

3. Place the file in the `data/` folder:

```bash
# If you downloaded a zip, extract it first:
unzip glove.6B.50d.txt.zip

# Then copy the file to the data directory:
cp glove.6B.50d.txt data/
```

Your `data/` folder should look like this:

```
data/
├── glove.6B.50d.txt   ✅
├── input-Copy1.txt
└── input.txt
```

---

trained on 6 billion tokens with 50-dimensional vectors.

Each word is represented as a point in ℝ⁵⁰ where geometric relationships reflect semantic ones — the classic example being:

```
vector(king) − vector(man) + vector(woman) ≈ vector(queen)
```

---
