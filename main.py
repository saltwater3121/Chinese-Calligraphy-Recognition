#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chinese Calligraphy Recognition - Main Entry Point
"""

import argparse
import sys
from pathlib import Path

# Add src to path so we can import callivision
sys.path.insert(0, str(Path(__file__).parent / "src"))

from callivision.train.train import main as train_main

def main():
    parser = argparse.ArgumentParser(
        description="Chinese Calligraphy Recognition Training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to config file (default: configs/baseline.yaml)",
    )
    args = parser.parse_args()
    
    print(f"Loading config from: {args.config}")
    train_main(args.config)

if __name__ == "__main__":
    main()
