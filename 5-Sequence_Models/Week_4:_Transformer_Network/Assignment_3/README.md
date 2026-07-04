# Named Entity Recognition (NER) - Transformer Application

## Setup

### 1. Create Environment

```bash
conda create -n tf_26 python=3.7
conda activate tf_26
conda install conda-forge::transformers==4.15.0
conda install conda-forge::pytorch==1.7.1
```

### 2. Download Pretrained Model

Download `tf_model.h5` from Hugging Face:

```
https://huggingface.co/google-bert/bert-base-cased/resolve/main/tf_model.h5?download=true
```

Move the downloaded file into the `model/` directory.

## Directory Structure

```
.
├── 28_Transformer_application_Named_Entity_Recognition.ipynb
├── model
│   ├── config.json
│   └── tf_model.h5
├── ner.json
└── tokenizer
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt

3 directories, 7 files
```

## Usage

Activate the environment and launch the notebook:

```bash
conda activate tf_26
jupyter notebook 28_Transformer_application_Named_Entity_Recognition.ipynb
```
