# asl-cnn
Lightweight CNN trained on the ASL-Alphabet dataset (29 hand-sign classes).   End-to-end notebook + scripts covering: data pipeline → model architecture → training → evaluation → ablation studies.   Achieves 86 % validation accuracy in 15 epochs with just 1.2 M parameters—perfect for learning the basics of image classification.


[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## TL;DR
Train a **1.2 M-parameter CNN** to classify **29 ASL hand-gesture classes**  
→ **86 % validation accuracy** in **15 epochs**.

| Metric | Value |
|--------|-------|
| Validation Accuracy | 86.0 % |
| Training Accuracy | 99.1 % |
| Parameters | 1.18 M |
| Dataset | [ASL-Alphabet (Kaggle)](https://www.kaggle.com/grassknoted/asl-alphabet) |

---

## Quick Start
```bash
git clone https://github.com/francesconb/asl-cnn.git
cd asl-cnn
pip install -r requirements.txt
python train.py              # or open notebooks/ASL_CNN.ipynb
python evaluate.py --weights checkpoints/best.h5
```
## Repository Layout
```bash
asl-cnn/
├── notebooks/
│   └── ASL_CNN.ipynb    # step-by-step walk-through
├── src/
│   ├── model.py         # CNN architecture
│   ├── train.py         # training script
│   └── utils.py         # preprocessing helpers
├── outputs/             # plots & logs
├── requirements.txt
└── README.md
