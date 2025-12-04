"""
Dataset loading and preprocessing for CyberSecLLM.

This module handles loading the cybersecurity dataset from Hugging Face
and preprocessing it for T5 fine-tuning.
"""

from typing import Dict, Any, Optional
from datasets import load_dataset, Dataset
from transformers import T5Tokenizer

from .config import Config, format_prompt


class CyberSecDataset:
    """Handles loading and preprocessing of the cybersecurity dataset."""
    
    def __init__(self, config: Config, tokenizer: T5Tokenizer):
        """
        Initialize the dataset handler.
        
        Args:
            config: Configuration object with dataset parameters
            tokenizer: T5 tokenizer for text preprocessing
        """
        self.config = config
        self.tokenizer = tokenizer
        self._dataset: Optional[Dataset] = None
        self._processed_dataset: Optional[Dataset] = None
    
    def load(self) -> Dataset:
        """
        Load the cybersecurity dataset from Hugging Face.
        
        Returns:
            The loaded dataset
        """
        print(f"Loading dataset: {self.config.dataset_name}")
        dataset = load_dataset(self.config.dataset_name)
        train_data = dataset["train"]
        
        # Optionally limit number of samples
        if self.config.train_samples:
            train_data = train_data.select(range(min(self.config.train_samples, len(train_data))))
        
        # Filter out samples with specific prefix (low quality)
        if self.config.filter_prefix:
            original_len = len(train_data)
            train_data = train_data.filter(
                lambda x: not (x["instruction"] or "").strip().startswith(self.config.filter_prefix)
            )
            filtered_count = original_len - len(train_data)
            print(f"Filtered {filtered_count} samples with prefix: '{self.config.filter_prefix[:50]}...'")
        
        self._dataset = train_data
        print(f"Loaded {len(train_data)} training samples")
        return train_data
    
    def preprocess(self, dataset: Optional[Dataset] = None) -> Dataset:
        """
        Preprocess the dataset for T5 training.
        
        Args:
            dataset: Optional dataset to preprocess. Uses loaded dataset if None.
            
        Returns:
            Preprocessed dataset ready for training
        """
        if dataset is None:
            dataset = self._dataset
        if dataset is None:
            raise ValueError("No dataset loaded. Call load() first or provide a dataset.")
        
        print("Preprocessing dataset...")
        processed = dataset.map(
            self._preprocess_example,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        self._processed_dataset = processed
        return processed
    
    def _preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a single example.
        
        Args:
            example: Raw example from the dataset
            
        Returns:
            Tokenized example with input_ids, attention_mask, and labels
        """
        question = example.get("instruction", "") or ""
        answer = example.get("output", "") or ""
        
        # Format the input prompt
        source = format_prompt(question)
        
        # Tokenize input
        model_inputs = self.tokenizer(
            source,
            truncation=True,
            max_length=self.config.max_input_length,
        )
        
        # Tokenize target (answer)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                answer,
                truncation=True,
                max_length=self.config.max_output_length,
            )["input_ids"]
        
        model_inputs["labels"] = labels
        return model_inputs
    
    @property
    def train_dataset(self) -> Dataset:
        """Get the processed training dataset."""
        if self._processed_dataset is None:
            raise ValueError("Dataset not preprocessed. Call preprocess() first.")
        return self._processed_dataset
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        if self._dataset is not None:
            return len(self._dataset)
        return 0


def get_dataset(config: Config, tokenizer: T5Tokenizer) -> Dataset:
    """
    Convenience function to load and preprocess the dataset.
    
    Args:
        config: Configuration object
        tokenizer: T5 tokenizer
        
    Returns:
        Preprocessed dataset ready for training
    """
    dataset_handler = CyberSecDataset(config, tokenizer)
    dataset_handler.load()
    return dataset_handler.preprocess()

