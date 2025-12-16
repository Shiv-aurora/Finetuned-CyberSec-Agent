# CyberSecLLM Methodology

This document details the methodology used to develop CyberSecLLM, including data preparation, training process, hyperparameter optimization, and deployment strategies.

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
6. [Model Quantization](#model-quantization)
7. [Deployment](#deployment)

---

## Problem Statement

### The Challenge

Standard language models, even powerful ones like GPT and T5, often struggle with domain-specific cybersecurity questions. Common issues include:

1. **Question Echoing**: Models repeat the question instead of answering
2. **Generic Responses**: Answers lack technical depth and domain terminology
3. **Hallucination**: Generating plausible-sounding but incorrect security information
4. **Inconsistency**: Different answers for similar questions

### Our Solution

Fine-tune a lightweight T5-small model on a curated cybersecurity dataset to:
- Provide accurate, domain-specific answers
- Use appropriate security terminology
- Maintain consistency across similar queries
- Be deployable on resource-constrained environments

---

## Dataset

### Source

We use the [Alpaca CyberSecurity Dataset 2.0](https://huggingface.co/datasets/Mohabahmed03/Alpaca_Dataset_CyberSecurity_2.0) from Hugging Face.

### Statistics

| Metric | Value |
|--------|-------|
| Total Samples | 159,217 |
| After Filtering | ~150,000 |
| Average Question Length | ~45 tokens |
| Average Answer Length | ~120 tokens |

### Topics Covered

The dataset spans comprehensive cybersecurity domains:

- **Network Security**: Firewalls, IDS/IPS, VPNs, network protocols
- **Web Security**: SQLi, XSS, CSRF, SSRF, authentication
- **Cryptography**: Symmetric/asymmetric encryption, hashing, PKI
- **Authentication**: Kerberos, OAuth, SAML, MFA
- **Threat Intelligence**: MITRE ATT&CK, TTPs, indicators
- **Malware Analysis**: Types, behavior, detection
- **Cloud Security**: AWS, Azure, GCP security
- **Compliance**: GDPR, PCI-DSS, SOC2

### Data Preprocessing

```python
# Filtering low-quality samples
prefix_filter = "Please provide detailed information about"
train_data = dataset["train"].filter(
    lambda x: not x["instruction"].startswith(prefix_filter)
)
```

**Why filter?** Some samples were templated questions that produced generic, unhelpful responses. Removing them improved model quality.

### Prompt Format

```
Answer the following cybersecurity question.

Question: {question}
Answer:
```

This structured format helps the model understand the task and produces more focused responses.

---

## Model Architecture

### Base Model: T5-Small

We chose T5-small for several reasons:

| Factor | T5-Small | Larger Alternatives |
|--------|----------|---------------------|
| Parameters | 60M | 220M-11B |
| Training Time | ~30 min | Hours to days |
| Inference Speed | Fast | Slower |
| Memory | 242 MB | GB+ |
| Quality | Good for Q&A | Better but overkill |

### Why T5?

1. **Text-to-Text Framework**: Natural fit for Q&A tasks
2. **Pre-trained on Diverse Data**: Good starting point
3. **Efficient**: Can run on consumer hardware
4. **Well-Documented**: Extensive HuggingFace support

---

## Training Process

### Hardware

- **GPU**: NVIDIA T4/V100 (Google Colab)
- **Memory**: 16GB GPU RAM
- **Training Time**: ~30 minutes for full dataset

### Training Configuration

```python
TrainingArguments(
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    max_input_length=384,
    warmup_ratio=0.03,
    weight_decay=0.01,
    fp16=True,  # Mixed precision training
)
```

### Training Curves

The model converges smoothly without significant overfitting:

- **Loss Reduction**: 2.5 → 0.8 over 3 epochs
- **No Validation Spike**: Indicating good generalization

### Generation Parameters

For inference, we use careful sampling to balance quality and diversity:

```python
GenerationConfig(
    max_new_tokens=128,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    no_repeat_ngram_size=3,
    repetition_penalty=1.1,
)
```

---

## Hyperparameter Optimization

### Grid Search

We systematically tested combinations of:

| Parameter | Values Tested | Best |
|-----------|---------------|------|
| Learning Rate | 3e-5, 1e-4, 3e-4 | **3e-5** |
| Batch Size | 2, 4, 8 | **8** |
| Epochs | 1, 2, 3 | **3** |
| Max Input Length | 128, 256, 384 | **384** |

### Results Summary

```
Best Configuration:
- Learning Rate: 3e-5 (lower = more stable)
- Batch Size: 8 (higher = better gradients)
- Epochs: 3 (more training = better quality)
- Max Input: 384 (capture full context)

Combined Score: 0.308 (highest achieved)
```

### Key Findings

1. **Lower Learning Rates Win**: 3e-5 consistently outperformed higher rates
2. **Larger Batches Help**: 8 > 4 > 2 for stable training
3. **More Epochs ≠ Overfitting**: 3 epochs still improved quality
4. **Context Length Matters**: 384 tokens captured important details

---

## Model Quantization

### INT8 Dynamic Quantization

To enable deployment on resource-constrained devices:

```python
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### Size Reduction

| Version | Size | Change |
|---------|------|--------|
| Original FP32 | 242 MB | Baseline |
| INT8 Quantized | 66 MB | **-73%** |

### Quality Impact

Quantization has minimal impact on output quality:

| Metric | Original | Quantized | Change |
|--------|----------|-----------|--------|
| Coherence | 0.450 | 0.445 | -1.1% |
| Perplexity | 1.189 | 1.195 | +0.5% |

---

## Deployment

### Hugging Face Hub

The model is uploaded to HuggingFace for easy access:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="./model",
    repo_id="username/cybersec-t5",
    repo_type="model"
)
```

### Hugging Face Spaces

Interactive demo using Gradio:

```python
import gradio as gr

demo = gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(label="Question"),
    outputs=gr.Textbox(label="Answer"),
)
demo.launch()
```

### Local Deployment Options

1. **Python API**: Use `src/model.py` for programmatic access
2. **CLI**: Run `scripts/evaluate.py` for batch processing
3. **REST API**: Wrap with FastAPI for production

---

## Lessons Learned

### What Worked

1. ✅ Structured prompt format improved consistency
2. ✅ Lower learning rates prevented overfitting
3. ✅ INT8 quantization maintained quality
4. ✅ Filtering low-quality data improved results

### What We'd Do Differently

1. 🔄 Try larger T5 variants (T5-base, T5-large)
2. 🔄 Experiment with LoRA for efficient fine-tuning
3. 🔄 Add retrieval-augmented generation (RAG)
4. 🔄 Create domain-specific evaluation benchmarks

---

## References

1. [T5 Paper](https://arxiv.org/abs/1910.10683): Exploring the Limits of Transfer Learning
2. [Hugging Face Transformers](https://huggingface.co/docs/transformers)
3. [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
4. [MITRE ATT&CK Framework](https://attack.mitre.org/)

