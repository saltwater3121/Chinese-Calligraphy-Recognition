#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference script for Chinese Calligraphy Recognition
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from callivision.config import load_cfg
from callivision.models.resnet import build_resnet18
from callivision.data.transforms import build_transforms


def predict(
    image_path: str,
    model_path: str,
    class_idx_path: str,
    img_size: int = 224,
    device: str = "cuda",
):
    """
    Predict the class of a single image.
    
    Args:
        image_path: Path to the input image
        model_path: Path to the trained model weights
        class_idx_path: Path to class_to_idx.json
        img_size: Image size for resizing
        device: Device to run inference on (cuda or cpu)
    
    Returns:
        Predicted class name and confidence score
    """
    device = torch.device(
        device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    
    # Load class mapping
    with open(class_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # Build model
    num_classes = len(class_to_idx)
    model = build_resnet18(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load and preprocess image
    tfm = build_transforms(img_size)
    image = Image.open(image_path).convert("RGB")
    image_tensor = tfm(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    pred_class = idx_to_class[pred_idx]
    
    return pred_class, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Inference for Chinese Calligraphy Recognition"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="runs/baseline/best.pt",
        help="Path to model weights (default: runs/baseline/best.pt)",
    )
    parser.add_argument(
        "--class-idx",
        type=str,
        default="runs/baseline/class_to_idx.json",
        help="Path to class_to_idx.json (default: runs/baseline/class_to_idx.json)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size (default: 224)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu, default: cuda)",
    )
    
    args = parser.parse_args()
    
    pred_class, confidence = predict(
        args.image,
        args.model,
        args.class_idx,
        args.img_size,
        args.device,
    )
    
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
