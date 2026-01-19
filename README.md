# callivision

Chinese Calligraphy Recognition using Deep Learning.

## Project Structure

```
.
├── configs/
│   └── baseline.yaml          # Configuration file
├── src/
│   └── callivision/
│       ├── __init__.py
│       ├── config.py          # Configuration loader
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py     # DataLoader builder
│       │   └── transforms.py  # Image transformations
│       ├── models/
│       │   ├── __init__.py
│       │   └── resnet.py      # ResNet18 model
│       └── train/
│           ├── __init__.py
│           └── train.py       # Training loop
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Chinese-Calligraphy-Recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Run the training script with the default configuration:
```bash
python main.py
```

Or specify a custom config file:
```bash
python main.py --config configs/baseline.yaml
```

### Configuration

Edit `configs/baseline.yaml` to customize:
- `seed`: Random seed for reproducibility
- `device`: cuda or cpu
- `data.root`: Path to dataset
- `data.img_size`: Image size for resizing
- `data.batch_size`: Batch size for training
- `model.name`: Model architecture
- `model.num_classes`: Number of classes
- `train.epochs`: Number of training epochs
- `train.lr`: Learning rate
- `train.weight_decay`: Weight decay for optimizer
- `train.save_dir`: Directory to save model checkpoints

### Dataset Structure

The dataset should be organized as follows:
```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
│       └── img3.jpg
└── val/
    ├── class1/
    │   └── img4.jpg
    └── class2/
        └── img5.jpg
```

## Features

- **ResNet18 Architecture**: Pre-configured deep learning model
- **Data Augmentation**: Random rotation and color jittering
- **Multi-GPU Support**: Automatic CUDA device detection
- **Checkpoint Saving**: Saves best model based on validation accuracy
- **Progress Tracking**: Uses tqdm for training progress visualization

## Output

Training results are saved in the directory specified by `train.save_dir`:
- `best.pt`: Best model weights
- `class_to_idx.json`: Mapping of class names to indices

## Requirements

- Python 3.7+
- PyTorch
- TorchVision
- OpenCV
- NumPy
- PyYAML
- scikit-learn
- tqdm
