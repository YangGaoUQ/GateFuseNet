# GateFuseNet: An Adaptive 3D Multimodal Neuroimaging Fusion Network for Parkinson's Disease Diagnosis

This repository contains the official implementation of **GateFuseNet**, an adaptive 3D multimodal MRI fusion network for automated Parkinson's Disease (PD) diagnosis. 

## 🔬 Abstract

Accurate Parkinson's Disease (PD) diagnosis based on MRI images remains challenging due to complicated symptom variability and pathological heterogeneity. Our proposed GateFuseNet addresses these challenges by combining:

- **QSM (Quantitative Susceptibility Mapping)**: Provides quantitative iron deposition information
- **T1-weighted images**: Delivers high-resolution anatomical boundaries  
- **ROI masks**: Pathological-aware region guidance

**Key Innovation**: Gated Fusion (GF) blocks with Adaptive Multimodal Fusion (AMF) and Channel-wise Gating (CWG) mechanisms for selective feature modulation.

## 🏆 Results

- **Accuracy**: 85.00%
- **AUC-ROC**: 92.06%
- **Precision**: 84.98%
- **Recall**: 86.06%
- **F1-score**: 85.48%

## 🏗️ Architecture

### Network Components

1. **Stem Module**: Three parallel 3D encoders for different modalities
2. **Fusion Modules**: CBAM-augmented bottleneck blocks with GF block
3. **Decision Module**: CBAM-dilated bottleneck blocks with classification head

### Key Features

- **Adaptive Multimodal Fusion (AMF)**: Learns modality-specific attention weights
- **Channel-wise Gating (CWG)**: Controls feature injection with learnable gates
- **Pathological-aware ROI guidance**: Emphasizes disease-relevant brain regions
- **Hierarchical fusion strategy**: Progressive integration across multiple stages

## 📁 Project Structure

```
GateFuseNet-PD-Classification/
├── models/
│   └── PD_AMF_c2.py             # Core GateFuseNet model implementation
├── dataset/
│   └── TripleDataset.py         # Triple modality dataset loader
├── utils/
│   ├── metrics.py               # Evaluation metrics
│   ├── losses.py                # Custom loss functions (Focal Loss)
│   ├── my_utils.py              # Utility functions
│   └── early_stopping.py        # Early stopping implementation
├── train_val.py                 # Training script with 5-fold CV
├── test.py                      # Model evaluation script
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
# Create conda environment
conda create -n gatefusenet python=3.9
conda activate gatefusenet

# Install dependencies
pip install torch torchvision torchaudio
pip install nibabel scipy scikit-learn
pip install matplotlib seaborn pandas
pip install torchio  # For 3D medical image augmentation
```

### Data Preparation

1. **Image Modalities Required**:

   - QSM images (Quantitative Susceptibility Mapping)
   - T1-weighted images
   - ROI segmentation masks

2. **Data Format**:

   - All images should be in NIfTI format (.nii or .nii.gz)
   - Standardized to 1×1×1 mm³ isotropic resolution
   - Center-cropped/zero-padded to 128×128×128 voxels

3. **Data Split File**:
   Create `train_val_splits.txt` with file paths for 5-fold cross-validation:

   ```
   fold_0_train: /path/to/qsm1.nii /path/to/t1_1.nii /path/to/seg1.nii 0
   fold_0_train: /path/to/qsm2.nii /path/to/t1_2.nii /path/to/seg2.nii 1
   ...
   fold_0_val: /path/to/qsm_val.nii /path/to/t1_val.nii /path/to/seg_val.nii 0
   ```

### Training

```bash
# Train with default parameters
python train_val.py --batch_size 4 --learning_rate 0.0002 --num_epochs 25 --num_folds 5 --modes_num 3

# Custom training parameters
python train_val.py \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --num_epochs 30 \
    --modes_num 3 \
    --data_split_file "your_train_val_splits.txt" \
    --save_dir "your_output_directory"
```

### Testing

```bash
# Evaluate trained models
python test.py --modes_num 3
```

### Key Parameters

- `--modes_num`: Number of input modalities (3 for QSM+T1w+ROI)
- `--batch_size`: Batch size for training (default: 4)
- `--learning_rate`: Initial learning rate (default: 0.0002)
- `--num_epochs`: Number of training epochs (default: 25)
- `--num_folds`: Number of cross-validation folds (default: 5)

## 🔧 Model Architecture Details

### GateFuseNet Core Components

1. **Stem Module**:

   ```python
   # Three parallel 3D encoders
   - QSM encoder: processes quantitative susceptibility data
   - T1w encoder: processes structural anatomical data  
   - ROI encoder: processes pathological region masks
   ```

2. **Gated Fusion (GF) Block**:

   ```python
   # Adaptive Multimodal Fusion (AMF)
   α_m = σ(BN(Conv3×3×3[x_QSM, x_T1, x_ROI]))
   
   # Channel-wise Gating (CWG) 
   f̂ = σ(θ) ⊙ f_fused
   ```

3. **Decision Module**:

   - Three CBAM-dilated bottleneck blocks
   - Global average pooling
   - Fully connected classification layer

### Loss Function

- **Focal Loss**: Addresses class imbalance by down-weighting easy samples
- Parameters: γ=2.0, α=0.5

## 📊 Experimental Results

### Performance Comparison

| Model           | Accuracy   | Precision  | Recall     | F1-score   | AUC        |
| --------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| ResNeXt         | 76.56%     | 80.63%     | 72.73%     | 75.97%     | 85.94%     |
| AG-SE-ResNeXt   | 76.88%     | 79.96%     | 75.76%     | 77.05%     | 88.31%     |
| DenseFormer-MoE | 81.25%     | 71.88%     | 88.46%     | 79.31%     | 90.84%     |
| **GateFuseNet** | **85.00%** | **84.98%** | **86.06%** | **85.48%** | **92.06%** |

### Ablation Studies

- **Multimodal advantage**: Triple input (T1w+QSM+ROI) significantly outperforms single/dual modality
- **Fusion mechanism**: Proposed gated fusion outperforms simple concatenation (+6.83% accuracy)
- **ROI guidance**: ROI branch placement achieves optimal performance

## 🧠 Clinical Interpretability

GateFuseNet provides interpretable results through:

- **Grad-CAM visualizations**: Show model attention on clinically relevant regions
- **Pathological focus**: Consistent attention on substantia nigra (SN) and globus pallidus (GP)
- **ROI-guided attention**: Leverages expert-defined anatomical priors

## 📂 Additional Resources

The source code is also available at: https://drive.google.com/drive/folders/1oA9NwmjQk7yA7P4NgeND-o7RqoVsqWr6?usp=sharing

