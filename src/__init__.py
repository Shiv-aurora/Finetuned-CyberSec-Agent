"""
CyberSecLLM: Fine-tuned T5 Model for Cybersecurity Question Answering

This package provides tools for training, evaluating, and deploying
a T5-small model fine-tuned on cybersecurity domain data.
"""

__version__ = "1.0.0"
__author__ = "Shivam Arora"

from .config import Config
from .model import CyberSecModel
from .data import CyberSecDataset
from .evaluate import Evaluator

__all__ = ["Config", "CyberSecModel", "CyberSecDataset", "Evaluator"]

