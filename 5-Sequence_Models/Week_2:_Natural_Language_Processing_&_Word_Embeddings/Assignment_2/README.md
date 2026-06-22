# Emojify — Word Vector Assignment (Week 2)

> **Course:** Deeplearning.ai — Sequence Models  
> **Week:** 2 — Natural Language Processing & Word Embeddings

Given a short sentence, this project predicts the most fitting emoji to append to it — using GloVe word embeddings and two model architectures: a simple averaged-embedding baseline (Emojify V1) and a stacked LSTM network (Emojify V2).

---

## Project Structure

```
Assignment_2/
├── 23_Emojify with Word Vector.ipynb   # Main notebook
├── emo_utils.py                        # Helper functions (data loading, emoji mapping)
├── test_utils.py                       # Autograder utilities
├── data/
│   ├── train_emoji.csv                 # Training set (sentences + emoji labels)
│   ├── test_emoji.csv                  # Test set
│   ├── emojify_data.csv                # Full dataset
│   ├── glove.6B.50d.txt                # ← GloVe embeddings (download required, see below)
│   └── fake_glove.6B.50d.txt           # Tiny stub used by the autograder
└── images/                             # Diagrams referenced in the notebook
```

---

## Setup

### 1. Download GloVe embeddings

The `glove.6B.50d.txt` file (~170 MB) is not included in the repository because of its size.

1. Go to the Kaggle dataset page:  
   **<https://www.kaggle.com/datasets/watts2/glove6b50dtxt>**
2. Download `glove.6B.50d.txt` (you need a free Kaggle account).
3. Place the file in the `data/` directory:

```bash
mv ~/Downloads/glove.6B.50d.txt data/
```

The notebook expects the file at exactly `data/glove.6B.50d.txt`.

---
