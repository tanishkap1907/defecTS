# defecTS — Semiconductor Defect Classifier

![accuracy](https://img.shields.io/badge/Accuracy-84.4%25-green)
![model](https://img.shields.io/badge/Model-EfficientNet--B0-blue)
![challenge](https://img.shields.io/badge/ASU%20×%20Intel-2026-red)

> Semiconductor Solutions Challenge 2026 — Problem A  
> Developed in partnership with Intel Corporation × Arizona State University

---

## What is defecTS?

defecTS is an AI-powered semiconductor wafer defect classifier built for the 
Semiconductor Solutions Challenge 2026. It classifies grayscale wafer images 
into 9 categories (8 defect types + good) using transfer learning on 
EfficientNet-B0.

**Key Achievement:** 84.4% overall classification accuracy on a severely 
imbalanced dataset (793x ratio between majority and minority classes).

---

## Problem Statement

In semiconductor manufacturing, labeled defect data is extremely scarce.
Traditional ML requires thousands of examples per class — but in reality:

| Class    | Images |
|----------|--------|
| good     | 7,135  |
| defect8  | 803    |
| defect10 | 674    |
| defect5  | 411    |
| defect9  | 319    |
| defect1  | 253    |
| defect2  | 178    |
| defect4  | 14     |
| defect3  | 9      |

This is a classic **small sample learning** problem.

---

## Our Solution

### Model Architecture
- **Base:** EfficientNet-B0 (pretrained on ImageNet)
- **Approach:** Transfer learning + gradual fine-tuning
- **Input:** 224×224 grayscale images (converted to 3-channel)
- **Output:** 9-class softmax prediction
- **Dropout:** 0.3 for regularization
- **Inference time:** ~50ms per image ✓

### Handling Class Imbalance
1. **Weighted CrossEntropyLoss** — rare class errors penalized more heavily
2. **Data Augmentation** — random flips, rotations, color jitter, affine transforms
3. **Gradual Layer Unfreezing** — freeze early layers first, unfreeze progressively

### Training Strategy
| Run | Key Change | Val Accuracy |
|-----|-----------|--------------|
| 1 | Baseline | 63.6% |
| 2 | Removed sampler, added dropout | 80.7% |
| 3 | Full fine-tuning, layer-wise LR | 83.9% |
| 4 | Cosine annealing + label smoothing | **84.4%** |

---

## Results

| Metric | Value |
|--------|-------|
| Overall Accuracy | **84.4%** |
| Target Accuracy | ~85% |
| Inference Time | ~50ms |
| Classes | 9 |
| Total Training Images | 9,796 |

### Per Class Accuracy
| Class | Accuracy | Images |
|-------|----------|--------|
| good | 88.6% | 7,135 |
| defect9 | 82.4% | 319 |
| defect1 | 78.7% | 253 |
| defect10 | 79.4% | 674 |
| defect5 | 68.2% | 411 |
| defect8 | 66.5% | 803 |
| defect2 | 65.7% | 178 |
| defect3 | 0.0% | 9 ⚠️ |
| defect4 | 0.0% | 14 ⚠️ |

> **Note:** defect3 and defect4 scored 0% due to insufficient data
> (9 and 14 images respectively). This is a data limitation, not a 
> model failure. More labeled samples would significantly improve these classes.

---

## Project Structure
```
defecTS/
├── data/                    # Dataset (not included in repo)
│   ├── defect1/
│   ├── defect2/
│   ├── defect3/
│   ├── defect4/
│   ├── defect5/
│   ├── defect8/
│   ├── defect9/
│   ├── defect10/
│   └── good/
├── train.py                 # Model training script
├── evaluate.py              # Evaluation + plot generation
├── explore.py               # Dataset exploration
├── app.py                   # Streamlit web application
├── best_model.pth           # Trained model weights
├── plot_confusion_matrix.png
├── plot_per_class_accuracy.png
├── plot_accuracy_vs_occurrence.png
└── README.md
```

---

## Setup & Installation

### Requirements
- Python 3.10 or 3.11 (recommended)
- Mac with Apple Silicon (MPS) or any GPU

### Install Dependencies
```bash
git clone https://github.com/YOURUSERNAME/defecTS.git
cd defecTS
python -m venv venv
source venv/bin/activate
pip install torch torchvision timm scikit-learn matplotlib seaborn streamlit pillow tqdm plotly pandas
```

### Dataset
Place the Intel-provided dataset in a folder called `data/` with one 
subfolder per class.

---

## Running the Project

### Train the model
```bash
python train.py
```

### Evaluate and generate plots
```bash
python evaluate.py
```

### Run the web app
```bash
streamlit run app.py
```

---

## Quick Demo (Terminal)
```python
import torch, timm, os
from torchvision import transforms
from PIL import Image

DEVICE = torch.device('mps')
CLASS_NAMES = ['defect1','defect10','defect2','defect3',
               'defect4','defect5','defect8','defect9','good']

model = timm.create_model('efficientnet_b0', pretrained=False, 
                           num_classes=9, drop_rate=0.3)
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
model.to(DEVICE).eval()

# Result:
# Actual: good       Predicted: good       Confidence: 100.0%
# Actual: defect1    Predicted: defect1    Confidence: 100.0%
# Actual: defect8    Predicted: defect8    Confidence: 100.0%
```

---

## Hardware Used
- **Device:** Apple MacBook (Apple Silicon M-series)
- **Backend:** PyTorch MPS (Metal Performance Shaders)
- **Training time:** ~3 minutes per epoch
- **Inference:** ~50ms per image

---

## Technologies
- PyTorch 2.10
- timm 1.0.25 (EfficientNet-B0)
- scikit-learn 1.8
- Streamlit
- Plotly
- Python 3.13

---

## Assumptions & Limitations
- Images are assumed to be grayscale wafer maps
- Model converts grayscale to 3-channel for EfficientNet compatibility
- defect3 and defect4 require more labeled data for reliable classification
- Model was trained and tested on Apple Silicon — may need adjustment for other hardware

---

## Future Work
- Collect more images for defect3 and defect4
- Implement Prototypical Networks for true few-shot learning
- Ensemble multiple models for higher accuracy
- Deploy on Intel hardware for production use
- Active learning pipeline for continuous improvement

---

## Competition
**Semiconductor Solutions Challenge 2026**  
Problem A: Small Sample Learning for Defect Classification  
Developed in partnership with Intel Corporation  
Arizona State University — Southwest Advanced Prototyping Hub

---

*Intel and the Intel logo are trademarks of Intel Corporation*