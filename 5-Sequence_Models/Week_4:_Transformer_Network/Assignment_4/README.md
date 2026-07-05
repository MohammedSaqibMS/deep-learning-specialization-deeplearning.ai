# Assignment 4: QA Transformer

A question-answering pipeline built with a BERT-based transformer, using both TensorFlow and PyTorch backends.

## Setup

### 1. Create the Environment

```bash
conda create -n tf_26 python=3.7
conda activate tf_26
conda install conda-forge::tensorflow==2.6.0 conda-forge::tensorflow-gpu==2.6.0
conda install conda-forge::transformers==4.15.0
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

### 2. Download Pretrained Model

Download `tf_model.h5` and `pytorch_model.bin` from Hugging Face:

```
https://huggingface.co/google-bert/bert-base-cased/resolve/main/tf_model.h5?download=true
https://huggingface.co/google-bert/bert-base-cased/resolve/main/pytorch_model.bin?download=true
```

Move the downloaded files into the `model/tensorflow` and `model/pytorch` directories respectively.

## Project Structure

Once set up, your project directory should look like this:

```
.
├── 29_QA_transformer.ipynb
├── data
│   ├── dataset_dict.json
│   ├── test
│   │   ├── dataset.arrow
│   │   ├── dataset_info.json
│   │   └── state.json
│   └── train
│       ├── dataset.arrow
│       ├── dataset_info.json
│       └── state.json
├── model
│   ├── pytorch
│   │   ├── config.json
│   │   └── pytorch_model.bin
│   └── tensorflow
│       ├── config.json
│       └── tf_model.h5
└── tokenizer
    ├── config.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt

8 directories, 16 files
```

## Usage

Open `29_QA_transformer.ipynb` in Jupyter (with the `tf_26` environment as the kernel) to run the pipeline.
