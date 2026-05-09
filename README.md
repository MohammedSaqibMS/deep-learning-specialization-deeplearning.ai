# Deep Learning Specialization - Programming Assignments

![Status: Under Development](https://img.shields.io/badge/Status-Under_Development-yellow)

This repository contains the programming assignments for the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by deeplearning.AI, taught by Andrew Ng. The specialization includes five comprehensive courses, each with hands-on assignments designed to help you gain practical experience with deep learning.

## Acknowledgements

These assignments are part of the Deep Learning Specialization by DeepLearning.AI on Coursera, taught by Andrew Ng. All credit for the course content goes to the creators of the specialization.

---

# ? ResNet50 ? Deep Residual Networks (Keras / TensorFlow)

> **Assignment:** Week 2 � Convolutional Neural Networks � Deep Learning Specialization (DeepLearning.AI)

---


## ?? Known Issue ? GPU Floating-Point Precision Mismatch

**Status: ? Will be fixed soon**

When running on **NVIDIA GPU** (tested on RTX 4070 Laptop, compute capability 8.9 with cuDNN 9.1), the `identity_block` unit test fails with:

```
AssertionError: Wrong values with training=False
```

**Root cause:** GPU and CPU use different floating-point operation ordering (due to cuDNN kernel selection and parallel reduction strategies), producing slightly different numerical results than the hardcoded expected values in `public_tests.py`. The difference is small (~0.03) but exceeds the `atol=1e-5` tolerance set by the test.

**Observed discrepancy example:**

| | CPU Output | GPU Output |
|---|---|---|
| `resume[1, 1, 0]` | `192.71236` | `192.6833` |
| `resume[2, 0, 0]` | `578.1371` | `577.92505` |

**Impact:** The model architecture, training logic, and grader cells are **unaffected**. This is purely a numerical tolerance issue in the local test utility. The model trains and evaluates correctly on GPU.

**Workaround (temporary):** Run the notebook on CPU by setting:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

Add this at the top of the notebook before importing TensorFlow.

**Fix planned:** The `public_tests.py` tolerance will be relaxed to accommodate GPU floating-point variance (`atol=1e-1` or relative tolerance), and/or the expected values will be recomputed on GPU.

---

*Made with ?? as part of the Deep Learning Specialization.*
