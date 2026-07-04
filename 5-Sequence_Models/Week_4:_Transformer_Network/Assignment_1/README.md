# Sequence Models — Week 4: Transformer Network (Assignment 1)

Implementation of the Transformer architecture (Vaswani et al., *Attention Is All You Need*) from the DeepLearning.AI **Sequence Models** course, built with TensorFlow.

## Environment Setup

```bash
conda create -n tf_24 python=3.7
conda activate tf_24
conda install conda-forge::tensorflow==2.4.0
```

Verify installation:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
# 2.4.0
```

## Topics Covered

- Positional Encoding
- Scaled Dot-Product Attention
- Multi-Head Attention
- Padding Mask & Look-Ahead Mask
- Encoder Layer & Encoder Stack
- Decoder Layer & Decoder Stack
- Full Transformer Model (Encoder-Decoder)

## Folder

```
Sequence Models/Week 4: Transformer Network/Assignment_1/
├── README.md
├── Transformer_Subclass_v1.ipynb   # main assignment notebook (or similarly named)
├── data/                            # datasets used in the assignment
└── images/                          # diagrams referenced in the notebook
```

## Usage

1. Activate the environment:
   ```bash
   conda activate tf_24
   ```
2. Launch Jupyter and open the assignment notebook:
   ```bash
   jupyter notebook
   ```

## Notes

- This assignment is part of the DeepLearning.AI Sequence Models course (Course 5 of the Deep Learning Specialization).
- Built and tested with TensorFlow 2.4.0 on Python 3.7.

## References

- Vaswani, A. et al. (2017). *Attention Is All You Need*.
- DeepLearning.AI — Sequence Models course, Week 4.
