# 🎨 Neural Style Transfer (Art Generation with Neural Networks)

This project implements Neural Style Transfer using a pre-trained VGG19 network. It blends a "content" image and a "style" image together using deep convolutional neural networks to generate a new, unique image that holds the content of the first and the artistic style of the second.

---

## 📥 Prerequisites: Downloading VGG19 Weights

Before running the notebook, you need the pre-trained VGG19 weights. To ensure the model loads correctly in this local environment, please follow these steps:

1. **Create Directory:** Create a new folder named `pretrained-model` in the root of your project directory.
2. **Download Weights:** Download the weights from Kaggle: [VGG19 Weights Dataset](https://www.kaggle.com/datasets/saksham219/vgg19-weights)
3. **Extract:** Extract the downloaded archive file.
4. **Locate File:** Find the specific weights file named `vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5`.
5. **Move File:** Paste/move this `.h5` file into the **`pretrained-model/`** folder you created in Step 1.

Your project structure should look like this:
```text
your_project_directory/
│
├── pretrained-model/
│   └── vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5
│
├── images/               # (Your content and style images)
├── Art_Generation_with_Neural_Style_Transfer.ipynb
└── README.md
