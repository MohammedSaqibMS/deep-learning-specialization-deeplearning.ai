# Operations on Word Vectors — Debiasing

> **Course:** Sequence Models — Week 2: Natural Language Processing & Word Embeddings  
> **Assignment:** 1

---

## Overview

This assignment explores **word embeddings** and how to work with them in practice. You'll load pre-trained GloVe vectors, measure semantic similarity using cosine similarity, and apply **debiasing algorithms** to reduce gender bias encoded in word representations.

Key concepts covered:

- Loading and querying pre-trained word vectors (GloVe)
- Cosine similarity as a measure of word relatedness
- Word analogy tasks (e.g., *man → woman* as *king → ?*)
- Neutralization — projecting bias out of gender-neutral words
- Equalization — balancing gendered word pairs around the gender-neutral axis

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

### 1. Clone / navigate to the project

```bash
cd "Sequence Models/Week 2: Natural Language Processing & Word Embeddings/Assignment_1"
```

### 2. Install dependencies

```bash
pip install numpy
```

> NumPy is the only external dependency. The notebook uses standard Python and the helper module `w2v_utils.py` included in the repo.

### 3. Download the GloVe embeddings

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

## Usage

Launch Jupyter and open the notebook:

```bash
jupyter notebook "22_Operations on Word Vectors - Debiasing.ipynb"
```

Run the cells in order. The notebook is self-contained and walks through each concept with explanations and exercises.

---

## What's Inside the Notebook

| Section | Description |
|---|---|
| **Load Word Vectors** | Read `glove.6B.50d.txt` into a word-to-vector dictionary |
| **Cosine Similarity** | Implement and test cosine similarity between word embeddings |
| **Word Analogies** | Solve analogy tasks using vector arithmetic |
| **Debiasing — Neutralize** | Remove gender bias from gender-neutral words (e.g., *doctor*, *engineer*) |
| **Debiasing — Equalize** | Ensure gendered pairs (e.g., *actor/actress*) differ only along the gender axis |

---

## Key Functions

| Function | File | Purpose |
|---|---|---|
| `read_glove_vecs()` | `w2v_utils.py` | Loads GloVe embeddings from file |
| `cosine_similarity()` | Notebook | Computes similarity between two vectors |
| `complete_analogy()` | Notebook | Solves word analogy puzzles |
| `neutralize()` | Notebook | Projects a word vector onto the gender-neutral subspace |
| `equalize()` | Notebook | Equalizes a pair of word vectors around the bias axis |

---

## Background: GloVe Embeddings

**GloVe** (Global Vectors for Word Representation) vectors encode semantic meaning as dense vectors by factorizing a word co-occurrence matrix over a large corpus. The `6B.50d` variant was trained on 6 billion tokens with 50-dimensional vectors.

Each word is represented as a point in ℝ⁵⁰ where geometric relationships reflect semantic ones — the classic example being:

```
vector(king) − vector(man) + vector(woman) ≈ vector(queen)
```

---

## References

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) — Pennington et al., Stanford NLP
- [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520) — Bolukbasi et al., 2016
- [deeplearning.ai — Sequence Models Course](https://www.coursera.org/learn/nlp-sequence-models)
