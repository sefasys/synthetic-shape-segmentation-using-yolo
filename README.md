# Neural Networks Practice Exam: Synthetic Shape Segmentation 

This repository contains the complete implementation for the "Synthetic Shape Segmentation" practice exam. The objective of this project is to train a neural network (U-Net) to detect and segment synthetically generated geometric shapes overlaid onto real-world images from the MS COCO 2017 dataset.

## 📂 Repository Structure

The project is modularized into three primary Notebooks for execution on Google Colab or a local Jupyter environment.

```
.
├── src/
│   ├── 1_Data_Creation.ipynb     # Downloads COCO, applies synthetic shapes, creates splits
│   ├── 2_Model_Training.ipynb    # U-Net architecture, Combined Loss, Model Training (AMP)
│   └── 3_Model_Testing.ipynb     # Evaluation, Ablation Experiments, Visualization
├── results/
│   ├── metrics.json              # Saved quantitative test metrics
│   ├── training_metrics.json     # Saved losses and IoU progression over epochs
│   └── figures/                  # Generated plots and visualization masks
├── requirements.txt              # Environment dependencies and versions
├── README.md                     # This file
└── Report.pdf                    # Practice exam report
```

## ⚙️ Dependencies & Setup

The code was developed and tested using **Python 3.11**. The complete list of required libraries is specified in `requirements.txt`. 

To install the dependencies locally, run:
```bash
pip install -r requirements.txt
```

### Key Libraries Used:
- `torch` (PyTorch)
- `torchvision` (COCO datasets & V2 Transforms)
- `segmentation-models-pytorch` (Pre-trained U-Net backbones)
- `pycocotools` (COCO annotation parsing)
- `Pillow`, `matplotlib`, `numpy` (Image processing and visualization)

## 🚀 Execution Instructions (Google Colab / Local)

The notebooks are designed to be executed sequentially. They utilize Google Drive by default for persistent storage (so that datasets and models do not need to be re-downloaded/re-trained if the runtime disconnects).

1. **`1_Data_Creation.ipynb`**:
   - Run all cells to download the `train2017` and `val2017` subsets of COCO.
   - The code automatically handles deterministic splitting (5000 Train / 1000 Val / 1000 Test) using fixed seeds to prevent data leakage.
   - Synthetically perturbed shapes (with random color, opacity, and rotation) are overlaid onto the images.
   - The dataset is saved locally/to Drive as a zipped archive.

2. **`2_Model_Training.ipynb`**:
   - Run all cells to initialize the **ResNet34 + U-Net** architecture.
   - The model is trained using Mixed Precision (AMP) and a Combined Loss (BCE + Dice).
   - Training takes approximately 20 minutes on an NVIDIA L4 GPU.
   - The best model weights are checkpointed as `best_segmentation_model.pth`.

3. **`3_Model_Testing.ipynb`**:
   - Run all cells to load the saved `best_segmentation_model.pth`.
   - The notebook evaluates the model against a heuristic Edge Detection baseline.
   - Test metrics (IoU, Dice, Precision, Recall) are computed and exported to `results/metrics.json`.
   - Ablation experiments (Pre-training vs Scratch, Augmentation vs No Augmentation) are executed here.
   - Final visual predictions (12 images) are plotted and saved.

## 📊 Results Summary

The U-Net model successfully learned to isolate synthetic shapes from complex real-world backgrounds, heavily outperforming standard heuristic techniques.

| Metric | Proposed U-Net | Edge Baseline |
| :--- | :---: | :---: |
| **Foreground IoU (Jaccard)** | 0.8356 | 0.0279 |
| **Dice Coefficient** | 0.8748 | --- |
| **Precision** | 0.8901 | --- |
| **Recall** | 0.9352 | --- |

Detailed visual analyses, including failure cases (color camouflaging and occlusions), are documented in the accompanying PDF report.
