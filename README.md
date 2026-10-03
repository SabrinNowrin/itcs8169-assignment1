# itcs8169-assignment1
The CNN Challenge

# ITCS 8169 Assignment 1: CNN Classification (16 classes)

## Final Result
| | |
|---|---|
| Model | ResNet-18 pretrained on ImageNet-1K |
| Validation accuracy | **93.54%** (best epoch 13 of 15) |
| Test accuracy | **92.50%** (evaluated once, on the final model only) |
| Checkpoint | [final_resnet18.pt](https://drive.google.com/file/d/1IulKS8ge4OIGy-EDSy9srA7OV_AqL47i/view?usp=sharing) |

## Validation Strategy
The 2,400 training images are split 80/20 into 1,920 training and 480 validation images (seed 0).
All design choices were made using validation accuracy. The checkpoint from the epoch with the
best validation accuracy is kept. The test set (400 images) was used only once, at the end.

## Final Recipe
- Input: 224×224 RGB, ImageNet normalization
- Augmentation (training only): random resized crop (70–100% of image) + horizontal flip
- Optimizer: Adam, learning rate 1e-4
- Batch size 64, 15 epochs, cross-entropy loss

## Files
| File | Purpose |
|---|---|
| `train.py` | Trains the final model and saves a checkpoint |
| `evaluate.py` | Tests a checkpoint on the test set (overall + per-class accuracy) |
| `*.ipynb` | Colab notebook used for the experiments |
| `requirements.txt` | Software versions |
| `AI_USAGE.md` | How AI coding tools were used |

## How to Reproduce
1. Download the dataset and place it as: from the link given in assignment
```
   data/train/<16 class folders>
   data/test/<16 class folders>
```
2. Download `final_resnet18.pt` from the link above into a `checkpoints/` folder.
3. Run:
```
   python train.py --data_root data
   python evaluate.py --checkpoint checkpoints/final_resnet18.pt --data_root data
```

**On Google Colab** (GPU on), with the files in Google Drive:
```
from google.colab import drive
drive.mount('/content/gdrive')
%cd /content/gdrive/MyDrive/ITCS8169/assignment1
!python train.py --data_root data
!python evaluate.py --checkpoint checkpoints/final_resnet18.pt --data_root data
```

`train.py` takes about 4 minutes on a Colab GPU. Results may vary slightly between runs because of GPU randomness.

## Environment
- Google Colab, GPU: v6e-1 TPU
- PyTorch 2.11.0, torchvision 0.24.0+cpu

- Random seed: 0

## Experiments
| Experiment | Validation Acc. | Observation |
|---|---|---|
| Baseline (starter TNet, grayscale, 64×64) | 49.79% | |
| + Color (RGB) | 51.88% | |
| + Augmentation (crop + flip) | 60.21% | |
| + Pretrained ResNet-18, 224×224 (final) | 93.75% | |
