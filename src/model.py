"""
Model loading and inference utilities for CyberSecLLM.

This module provides a clean interface for loading T5 models
(both pretrained and fine-tuned) and performing inference.
"""

from typing import List, Optional, Union
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .config import Config, format_prompt


class CyberSecModel:
    """
    Wrapper class for the CyberSecLLM T5 model.
    
    Provides easy-to-use methods for loading models and generating answers
    to cybersecurity questions.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[Config] = None,
        load_in_8bit: bool = False,
    ):
        """
        Initialize the CyberSecLLM model.
        
        Args:
            model_path: Path to fine-tuned model or HF model ID. 
                       If None, loads pretrained t5-small.
            config: Configuration object. Uses defaults if None.
            load_in_8bit: Whether to load the model in 8-bit quantization.
        """
        self.config = config or Config()
        self.device = self.config.device
        self.load_in_8bit = load_in_8bit
        
        # Load tokenizer
        tokenizer_path = model_path or self.config.model_name
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        
        # Load model
        print(f"Loading model from: {model_path or self.config.model_name}")
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_path or self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 and not load_in_8bit else torch.float32,
        )
        
        if not load_in_8bit:
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def generate(
        self,
        question: str,
        max_new_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate an answer to a single cybersecurity question.
        
        Args:
            question: The cybersecurity question to answer
            max_new_tokens: Maximum number of tokens to generate
            **kwargs: Additional generation parameters
            
        Returns:
            The generated answer
        """
        prompt = format_prompt(question)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_length,
        ).to(self.device)
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.config.max_output_length,
            "num_beams": self.config.num_beams,
            "do_sample": self.config.do_sample,
            "top_p": self.config.top_p,
            "temperature": self.config.temperature,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            "repetition_penalty": self.config.repetition_penalty,
            "early_stopping": True,
            **kwargs
        }
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def generate_batch(
        self,
        questions: List[str],
        batch_size: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        show_progress: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Generate answers for multiple questions in batches.
        
        Args:
            questions: List of cybersecurity questions
            batch_size: Number of questions to process at once
            max_new_tokens: Maximum tokens per answer
            show_progress: Whether to print progress
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated answers
        """
        batch_size = batch_size or self.config.eval_batch_size
        answers = []
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            prompts = [format_prompt(q) for q in batch_questions]
            
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_input_length,
            ).to(self.device)
            
            gen_kwargs = {
                "max_new_tokens": max_new_tokens or self.config.max_output_length,
                "num_beams": self.config.num_beams,
                "do_sample": self.config.do_sample,
                "top_p": self.config.top_p,
                "temperature": self.config.temperature,
                "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
                "repetition_penalty": self.config.repetition_penalty,
                "early_stopping": True,
                **kwargs
            }
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs
                )
            
            batch_answers = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            answers.extend(batch_answers)
            
            if show_progress:
                print(f"Generated {min(i + batch_size, len(questions))}/{len(questions)} answers")
        
        return answers
    
    def get_model_size(self) -> dict:
        """
        Get the model size in parameters and MB.
        
        Returns:
            Dictionary with 'parameters' and 'size_mb' keys
        """
        num_params = sum(p.numel() for p in self.model.parameters())
        size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        
        return {
            "parameters": num_params,
            "size_mb": size_bytes / (1024 * 1024),
        }
    
    def save(self, output_dir: str):
        """
        Save the model and tokenizer to a directory.
        
        Args:
            output_dir: Directory to save the model
        """
        print(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Model saved successfully")


def load_model(
    model_path: Optional[str] = None,
    quantized: bool = False,
) -> CyberSecModel:
    """
    Convenience function to load a CyberSecLLM model.
    
    Args:
        model_path: Path to model or HF model ID
        quantized: Whether to load in 8-bit quantization
        
    Returns:
        Loaded CyberSecModel instance
    """
    return CyberSecModel(model_path=model_path, load_in_8bit=quantized)

