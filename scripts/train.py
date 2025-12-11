#!/usr/bin/env python3
"""
CLI script for training CyberSecLLM models.

Usage:
    python scripts/train.py --output-dir ./models/my_model
    python scripts/train.py --epochs 5 --batch-size 4 --learning-rate 1e-4
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config
from src.train import CyberSecTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CyberSecLLM model on cybersecurity Q&A data"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="google-t5/t5-small",
        help="Base model name or path (default: google-t5/t5-small)"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate (default: 3e-5)"
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=384,
        help="Maximum input sequence length (default: 384)"
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=None,
        help="Number of training samples to use (default: all)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory for training outputs (default: ./outputs)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./models/cybersec-t5",
        help="Directory to save the trained model (default: ./models/cybersec-t5)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config
    config = Config(
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_input_length=args.max_input_length,
        train_samples=args.train_samples,
        output_dir=args.output_dir,
        model_save_dir=args.save_dir,
    )
    
    print("=" * 60)
    print("CyberSecLLM Training")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max input length: {config.max_input_length}")
    print(f"Device: {config.device}")
    print("=" * 60)
    
    # Train
    trainer = CyberSecTrainer(config)
    trainer.setup()
    metrics = trainer.train()
    trainer.save()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Model saved to: {config.model_save_dir}")
    print(f"Training metrics: {metrics}")


if __name__ == "__main__":
    main()

