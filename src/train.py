"""
Training pipeline for CyberSecLLM.

This module provides the training logic for fine-tuning T5 models
on the cybersecurity dataset.
"""

import os
from typing import Optional
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from .config import Config
from .data import get_dataset


class CyberSecTrainer:
    """Trainer class for fine-tuning T5 on cybersecurity data."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration. Uses defaults if None.
        """
        self.config = config or Config()
        self.model: Optional[T5ForConditionalGeneration] = None
        self.tokenizer: Optional[T5Tokenizer] = None
        self.trainer: Optional[Trainer] = None
    
    def setup(self):
        """Set up the model, tokenizer, and dataset for training."""
        print(f"Setting up training on device: {self.config.device}")
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.config.model_name
        ).to(self.config.device)
        
        print(f"Loaded model: {self.config.model_name}")
        
        # Load and preprocess dataset
        self.train_dataset = get_dataset(self.config, self.tokenizer)
        
        # Create data collator
        self.data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=100,
            save_total_limit=1,
            save_strategy="epoch",
            fp16=self.config.fp16,
            report_to="none",
            dataloader_num_workers=2,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
        )
        
        print("Training setup complete")
    
    def train(self) -> dict:
        """
        Run the training loop.
        
        Returns:
            Training metrics dictionary
        """
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup() first.")
        
        print("Starting training...")
        print(f"  - Learning rate: {self.config.learning_rate}")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - Epochs: {self.config.epochs}")
        print(f"  - Max input length: {self.config.max_input_length}")
        
        result = self.trainer.train()
        
        print("Training complete!")
        return result.metrics
    
    def save(self, output_dir: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            output_dir: Directory to save the model. Uses config default if None.
        """
        save_dir = output_dir or self.config.model_save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving model to {save_dir}")
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        print("Model saved successfully")


def train_model(
    config: Optional[Config] = None,
    output_dir: Optional[str] = None,
) -> T5ForConditionalGeneration:
    """
    Convenience function to train a CyberSecLLM model.
    
    Args:
        config: Training configuration
        output_dir: Where to save the trained model
        
    Returns:
        The trained model
    """
    trainer = CyberSecTrainer(config)
    trainer.setup()
    trainer.train()
    
    if output_dir:
        trainer.save(output_dir)
    
    return trainer.model

