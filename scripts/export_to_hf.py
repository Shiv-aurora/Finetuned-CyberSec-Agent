#!/usr/bin/env python3
"""
Script to export CyberSecLLM model to Hugging Face Hub.

Usage:
    python scripts/export_to_hf.py --model-path ./models/cybersec-t5 --repo-name username/cybersec-t5
    
Prerequisites:
    1. Install huggingface_hub: pip install huggingface_hub
    2. Login to HF: huggingface-cli login
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


MODEL_CARD_TEMPLATE = '''---
language:
- en
license: mit
tags:
- cybersecurity
- t5
- question-answering
- text2text-generation
datasets:
- Mohabahmed03/Alpaca_Dataset_CyberSecurity_2.0
metrics:
- perplexity
pipeline_tag: text2text-generation
---

# CyberSecLLM: Fine-tuned T5 for Cybersecurity Q&A

A T5-small model fine-tuned on 159K+ cybersecurity question-answer pairs for domain-specific knowledge retrieval.

## Model Description

This model is a fine-tuned version of [google-t5/t5-small](https://huggingface.co/google-t5/t5-small) on the [Alpaca CyberSecurity Dataset](https://huggingface.co/datasets/Mohabahmed03/Alpaca_Dataset_CyberSecurity_2.0).

### Key Improvements Over Base T5

| Metric | Pre-trained | Fine-tuned | Improvement |
|--------|-------------|------------|-------------|
| Coherence | 0.049 | 0.450 | **+818%** |
| Combined Score | -0.054 | 0.202 | **+474%** |
| Perplexity | 1.25 | 1.19 | **-5%** |

## Usage

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("{repo_name}")
tokenizer = T5Tokenizer.from_pretrained("{repo_name}")

# Generate answer
question = "What is SQL injection and how can it be prevented?"
prompt = f"Answer the following cybersecurity question.\\n\\nQuestion: {{question}}\\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    no_repeat_ngram_size=3,
    repetition_penalty=1.1
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

## Training Details

- **Base Model**: google-t5/t5-small (60M parameters)
- **Dataset**: 159,217 cybersecurity Q&A pairs
- **Training**: 3 epochs, batch size 8, learning rate 3e-5
- **Hardware**: NVIDIA GPU with FP16 mixed precision

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 3e-5 |
| Batch Size | 8 |
| Epochs | 3 |
| Max Input Length | 384 |
| Warmup Ratio | 0.03 |
| Weight Decay | 0.01 |

## Evaluation Metrics

The model was evaluated using 11 custom metrics:

1. **Coherence Score** - Semantic similarity between consecutive sentences
2. **Hallucination Score** - Model confidence + repetition analysis
3. **Perplexity** - Language modeling quality
4. **Repetition Score** - N-gram repetition detection
5. **Prompt Copying Score** - Question overlap in answers
6. **Semantic Richness** - Vocabulary diversity

## Limitations

- Optimized for cybersecurity domain; may not generalize well to other topics
- Small model size (60M params) limits complex reasoning
- Generated answers should be verified by security professionals

## Citation

```bibtex
@misc{{cybersecllm2024,
  title={{CyberSecLLM: Fine-tuned T5 for Cybersecurity Question Answering}},
  author={{Shivam Arora}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/shiv-aurora/cybersec-t5-small}}
}}
```

## License

MIT License
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export CyberSecLLM model to Hugging Face Hub"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        required=True,
        help="Hugging Face repo name (e.g., username/cybersec-t5)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload CyberSecLLM fine-tuned model",
        help="Commit message for the upload"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Install it with: pip install huggingface_hub")
        sys.exit(1)
    
    print("=" * 60)
    print("Exporting to Hugging Face Hub")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Repository: {args.repo_name}")
    print("=" * 60)
    
    # Verify model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    # Create model card
    model_card_content = MODEL_CARD_TEMPLATE.format(repo_name=args.repo_name)
    model_card_path = os.path.join(args.model_path, "README.md")
    
    with open(model_card_path, "w") as f:
        f.write(model_card_content)
    print(f"Created model card: {model_card_path}")
    
    # Create repository
    api = HfApi()
    try:
        create_repo(
            repo_id=args.repo_name,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"Repository created/verified: {args.repo_name}")
    except Exception as e:
        print(f"Warning: Could not create repo: {e}")
    
    # Upload
    print("\nUploading model files...")
    api.upload_folder(
        folder_path=args.model_path,
        repo_id=args.repo_name,
        repo_type="model",
        commit_message=args.commit_message,
    )
    
    print("\n" + "=" * 60)
    print("Upload Complete!")
    print("=" * 60)
    print(f"Model available at: https://huggingface.co/{args.repo_name}")


if __name__ == "__main__":
    main()

