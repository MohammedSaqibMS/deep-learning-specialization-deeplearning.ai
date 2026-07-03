# Assignment 2: Embedding + Positional Encoding

This project is part of **Week 4: Transformer Network** (Sequence Models course).

## Setup

### 1. Download GloVe embeddings

This project uses the **GloVe 6B 100d** word embeddings.

1. Download the file from Kaggle:
   [glove6b100dtxt](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt)
2. Extract the downloaded archive if needed to get `glove.6B.100d.txt`.
3. Create a `glove` folder in the project root (if it doesn't already exist) and move the file into it:

   ```bash
   mkdir -p glove
   mv ~/Downloads/glove.6B.100d.txt glove/
   ```

### 2. Verify project structure

After moving the file, your project structure should look like this:

```
.
├── Embedding_plus_Positional_encoding.ipynb
├── glove
│   └── glove.6B.100d.txt
└── preprocessing.png

2 directories, 3 files
```

You can verify this yourself by running:

```bash
tree
```

## Usage

Open `Embedding_plus_Positional_encoding.ipynb` in Jupyter/Anaconda and run the cells. The notebook expects the GloVe embeddings to be located at `glove/glove.6B.100d.txt`.
