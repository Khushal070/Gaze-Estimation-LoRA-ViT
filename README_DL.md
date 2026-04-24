# Attention-Based Gaze Estimation with Adaptive Personalization

**Authors:** Khushal Trivedi, Ansh Soni  
**Course:** CS 7150 - Deep Learning, Spring 2026  
**Institution:** Khoury College of Computer Sciences, Northeastern University

---

## Overview

This project implements an appearance-based gaze estimation system using a pure Vision Transformer (DeiT-Small) backbone, with parameter-efficient personalization via LoRA adapters. Our method achieves a 30% relative error reduction over the baseline using only 9 calibration samples per user and updating just 0.56% of model parameters.

---

## Submission Contents

### 1. Main Notebook
`Khushal_Trivedi_Ansh_Soni_Attention_Based_Gaze_Estimation.ipynb`

Contains the complete pipeline:
- Dataset preparation and face normalization
- DeiT-Small + gaze head architecture
- Baseline training (3-fold leave-one-person-out)
- Personalization experiments (head-only and LoRA)
- Attention rollout visualization
- Interactive web demo

### 2. Trained Model Weights

Hosted on Google Drive (publicly accessible):

**Link:** [gazeformer_p folder](https://drive.google.com/drive/folders/1aPyEAhRHdlqxXY2e59dgn8uAAeSLx6na?usp=sharing)

Folder structure:

    gazeformer_p/
    ├── checkpoints/   - trained baseline model weights (.pt files)
    ├── data/          - normalized MPIIFaceGaze HDF5 files
    ├── demo/          - demo outputs and sample images
    └── results/       - visualization outputs (plots, attention maps, logs)

---

## How to Run

### Quick Demo (Gradio Web UI)

1. Open the notebook in Google Colab
2. Run these cells in order:
   - Environment Setup (installs dependencies)
   - Mount Google Drive & Configuration
   - Model Architecture (GazeFormer class)
   - Interactive Web UI (launches Gradio)
3. Click the Gradio public link that appears to open the demo
4. Grant webcam permission and the demo will run in real-time

**Total setup time:** approximately 2 minutes

### Full Reproduction

To reproduce all results from scratch:

1. Run all cells in order from Cell 1 to Cell 39
2. Dataset preparation requires the raw MPIIFaceGaze dataset (see Dataset section)
3. Each LOPO fold takes approximately 1 hour on an NVIDIA A100 GPU
4. Personalization experiments take under 1 minute per subject

### Configuration Paths

Update paths in **Cell 3** to match your Drive structure:

```python
PROJECT_DIR = Path('/content/drive/MyDrive/gazeformer_p')
CHECKPOINT_DIR = PROJECT_DIR / 'checkpoints'
RESULTS_DIR = PROJECT_DIR / 'results'
```

---

## Key Results

| Method              | Mean Angular Error | Trainable Params | % of Model |
|---------------------|-------------------|------------------|------------|
| Generic Baseline    | 7.96°             | -                | -          |
| Head-only Personalization | 5.59°       | 50,306           | 0.23%      |
| **LoRA Personalization (Ours)** | **5.56°** | **124,034** | **0.56%** |

**Relative error reduction:** 30.3%

### Per-Subject Breakdown

| Subject       | Baseline | LoRA  | Improvement |
|---------------|----------|-------|-------------|
| p00 (easy)    | 3.86°    | 3.60° | +0.26°      |
| p07 (medium)  | 7.94°    | 7.76° | +0.18°      |
| p14 (hard)    | 12.10°   | 5.31° | **+6.78°**  |

**Key Finding:** Personalization benefit scales with baseline difficulty. The hardest subject (p14) sees a 56% error reduction, while easy subjects see minimal gains.

---

## Dependencies

- Python 3.10+
- PyTorch 2.x with CUDA support
- timm (DeiT-Small backbone loader)
- peft (LoRA adapter library)
- h5py, scipy (data I/O)
- opencv-python, mediapipe (face detection and normalization)
- gradio (web UI)

All dependencies install automatically via the first notebook cell.

---

## Hardware

Trained and tested on Google Colab Pro with a single NVIDIA A100 GPU (40GB).

---

## Dataset

**MPIIFaceGaze**  
https://www.perceptualui.org/research/datasets/MPIIFaceGaze/


---

## Contact

For questions or reproduction issues, contact:

- Khushal Trivedi - trivedi.kh@northeastern.edu
- Ansh Soni - soni.ansh@northeastern.edu