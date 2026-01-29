# OmniWaterMask Training

Training code for the deep learning model used in [OmniWaterMask](https://github.com/DPIRD-DMA/OmniWaterMask) - a Python library for detecting water bodies in satellite and aerial imagery.

## Model Architecture

- **Architecture**: U-Net segmentation model
- **Input**: 4 bands (Red, Green, Blue, NIR)
- **Output**: 2 classes (water / not water)

## Training Data

The model is trained on two datasets:

1. **FLAIR Dataset** - French aerial imagery patches containing water bodies
2. **S1S2-Water Dataset** - Sentinel-1/Sentinel-2 water body dataset

## Installation

```bash
uv sync
```

## Notebooks

### Training
- [Train OWM.ipynb](https://github.com/DPIRD-DMA/OmniWaterMask-training/blob/main/Train%20OWM.ipynb) - Main training notebook

### Dataset Preparation

**FLAIR:**
- [Download FLAIR.ipynb](https://github.com/DPIRD-DMA/OmniWaterMask-training/blob/main/get%20datasets/FLAIR/Download%20FLAIR.ipynb) - Download the FLAIR dataset
- [Mosaic and split FLAIR.ipynb](https://github.com/DPIRD-DMA/OmniWaterMask-training/blob/main/get%20datasets/FLAIR/Mosaic%20and%20split%20FLAIR.ipynb) - Mosaic and split FLAIR scenes

**S1S2-Water:**
- [Download S1S2.ipynb](https://github.com/DPIRD-DMA/OmniWaterMask-training/blob/main/get%20datasets/S1S2%20Water/Download%20S1S2.ipynb) - Download S1S2 dataset
- [Split S1S2.ipynb](https://github.com/DPIRD-DMA/OmniWaterMask-training/blob/main/get%20datasets/S1S2%20Water/Split%20S1S2.ipynb) - Split into training patches

## Training

Key training features:

- Custom augmentations for remote sensing data (rotation, flip, resampling, random cropping)
- Dynamic Z-score normalization
- Distance transform weighted loss for improved boundary detection
- Gradient accumulation for larger effective batch sizes
- BF16 mixed precision training

## Model Export

Trained models are exported in multiple formats:
- PyTorch full model (`.pth`)
- PyTorch state dict (`.pth`)
- Safetensors (`.safetensors`)

## License

MIT License
