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

### 1. Clone / open the project

```bash
cd "~/anaconda_projects/Sequence Models/Week 2: Natural Language Processing & Word Embeddings/Assignment_2"
```

### 2. Install dependencies

The notebook was tested with the following versions. Pinning TensorFlow to 2.6.0 avoids Keras API compatibility issues with the course autograder.

```bash
pip install tensorflow==2.6.0 numpy pandas emoji
```

> **Note:** If you are on an Apple Silicon Mac, substitute `tensorflow-macos==2.6.0`.

### 3. Download GloVe embeddings

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

## Models

### Emojify V1 — Averaged Embeddings + Softmax

A simple baseline that averages the GloVe vectors for all words in a sentence and feeds the result into a softmax classifier. Fast to train, but ignores word order.

### Emojify V2 — Stacked LSTM

A Keras sequential model with:

| Layer | Details |
|---|---|
| Embedding | Pretrained GloVe 50-d, frozen |
| LSTM (1) | 128 units, `return_sequences=True` |
| Dropout | 0.5 |
| LSTM (2) | 128 units |
| Dropout | 0.5 |
| Dense | 5 units, softmax activation |

Captures word order and context, which noticeably improves accuracy on ambiguous sentences.

---

## Usage

Launch Jupyter and open the notebook:

```bash
jupyter notebook "23_Emojify with Word Vector.ipynb"
```

Run all cells top-to-bottom. The notebook walks through:

1. Exploring the dataset
2. Building and training Emojify V1
3. Analysing misclassified examples
4. Building and training Emojify V2 with Keras
5. Evaluating on the test set
6. Running your own sentences through the model

---

## Dataset

`train_emoji.csv` and `test_emoji.csv` contain short English sentences paired with one of five emoji labels:

| Label | Emoji |
|---|---|
| 0 | ❤️ |
| 1 | ⚾ |
| 2 | 😄 |
| 3 | 😞 |
| 4 | 🍴 |

---

## Acknowledgements

- **GloVe embeddings** — Pennington et al., Stanford NLP Group  
- **Course material** — [Deeplearning.ai Sequence Models](https://www.coursera.org/learn/nlp-sequence-models)  
- **Original assignment** — Andrew Ng, Kian Katanforoosh, Younes Bensouda Mourri
