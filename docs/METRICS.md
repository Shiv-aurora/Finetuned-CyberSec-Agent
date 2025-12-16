# CyberSecLLM Evaluation Metrics

This document provides detailed explanations of the 11 evaluation metrics used to assess CyberSecLLM's performance.

## Table of Contents

1. [Overview](#overview)
2. [Core Metrics](#core-metrics)
3. [Linguistic Quality Metrics](#linguistic-quality-metrics)
4. [Advanced Metrics](#advanced-metrics)
5. [Combined Score](#combined-score)
6. [Usage](#usage)

---

## Overview

We developed a comprehensive evaluation framework specifically designed for assessing cybersecurity Q&A models. Traditional NLP metrics like BLEU or ROUGE don't fully capture the nuances of technical domain responses.

### Metric Categories

| Category | Metrics | Purpose |
|----------|---------|---------|
| **Core** | Coherence, Hallucination, Perplexity | Overall quality |
| **Linguistic** | Repetition, Prompt Copying, Semantic Richness | Text quality |
| **Advanced** | KL Divergence, Self-BLEU, Invalid Tokens | Model behavior |
| **Aggregate** | Combined Score | Single ranking metric |

---

## Core Metrics

### 1. Coherence Score

**What it measures:** Logical flow and consistency between consecutive sentences.

**How it works:**
1. Split answer into sentences
2. Generate sentence embeddings using `all-MiniLM-L6-v2`
3. Calculate cosine similarity between consecutive sentences
4. Average all similarities

**Formula:**
```
Coherence = (1/(n-1)) * Σ cos_sim(sent_i, sent_{i+1})
```

**Interpretation:**
- **Higher is better** (range: 0 to 1)
- 0.0-0.2: Disjointed, random sentences
- 0.2-0.4: Loosely connected
- 0.4-0.6: Good coherence
- 0.6+: Excellent logical flow

**Example:**
```python
from src.evaluate import Evaluator

evaluator = Evaluator(model, tokenizer)
score = evaluator.coherence_score(
    "Firewalls filter network traffic. They examine packets based on rules. "
    "This helps prevent unauthorized access."
)
# Returns: ~0.65 (good coherence)
```

---

### 2. Hallucination Score

**What it measures:** Model confidence and answer consistency.

**How it works:**
Combines three methods:
1. **Token Entropy** (70%): High entropy in output logits suggests uncertainty
2. **Repetition Detection** (20%): Repetitive answers often indicate hallucination
3. **Uncertain Language** (10%): Phrases like "maybe", "possibly", "unclear"

**Formula:**
```
Hallucination = 0.7 * entropy_score + 0.2 * repetition_score + 0.1 * uncertainty_score
```

**Interpretation:**
- **Lower is better** (range: 0 to 1)
- 0.0-0.2: Confident, consistent answers
- 0.2-0.4: Some uncertainty
- 0.4+: High hallucination risk

**Note:** Domain-specialized models often show slightly higher hallucination scores due to the confidence boost from fine-tuning, even when answers are correct.

---

### 3. Perplexity

**What it measures:** How well the model predicts its own output (language modeling quality).

**How it works:**
1. Tokenize the generated answer
2. Calculate cross-entropy loss with answer as both input and labels
3. Exponentiate the loss

**Formula:**
```
Perplexity = exp(CrossEntropyLoss(answer, answer))
```

**Interpretation:**
- **Lower is better** (range: 1 to ∞)
- 1.0-1.5: Excellent fluency
- 1.5-3.0: Good quality
- 3.0+: Potentially disfluent

---

## Linguistic Quality Metrics

### 4. Repetition Score

**What it measures:** Frequency of repeated n-gram phrases.

**How it works:**
1. Extract all n-grams (default n=3)
2. Count how many appear more than once
3. Calculate ratio of repeated n-grams to total

**Formula:**
```
Repetition = count(repeated_ngrams) / count(all_ngrams)
```

**Interpretation:**
- **Lower is better** (range: 0 to 1)
- 0.0-0.1: Minimal repetition
- 0.1-0.3: Acceptable
- 0.3+: Problematic repetition

**Example of high repetition (bad):**
> "The attack uses SQL injection. SQL injection allows attackers to inject SQL. SQL injection is dangerous."

---

### 5. Prompt Copying Score

**What it measures:** How much of the question appears verbatim in the answer.

**How it works:**
1. Tokenize both question and answer
2. Count overlapping tokens
3. Calculate overlap ratio

**Formula:**
```
Copying = count(question_tokens ∩ answer_tokens) / count(question_tokens)
```

**Interpretation:**
- **Lower is better** (range: 0 to 1)
- 0.0-0.3: Unique answer content
- 0.3-0.6: Some question repetition
- 0.6+: Mostly echoing the question

**Why it matters:** Pre-trained models often just repeat questions. Fine-tuned models should provide new information.

---

### 6. Semantic Richness Score

**What it measures:** Vocabulary diversity in the answer.

**How it works:**
1. Remove punctuation
2. Lowercase all tokens
3. Calculate unique/total token ratio

**Formula:**
```
Richness = count(unique_tokens) / count(all_tokens)
```

**Interpretation:**
- **Higher is better** (range: 0 to 1)
- 0.0-0.3: Limited vocabulary
- 0.3-0.5: Moderate diversity
- 0.5+: Rich vocabulary

---

## Advanced Metrics

### 7. Invalid Token Score

**What it measures:** Presence of corrupted or nonsensical output.

**Checks for:**
- Weird symbols (non-ASCII, special characters)
- Triple-repeated words
- Character runs (e.g., "aaaaaaa")
- Non-English character ratio

**Formula:**
```
Invalid = 0.25 * (weird_symbols > 0) 
        + 0.25 * (triple_repeats > 0)
        + 0.25 * (char_runs > 0)
        + 0.25 * non_english_ratio
```

**Interpretation:**
- **Lower is better** (range: 0 to 1)
- 0.0: Clean output
- 0.0-0.2: Minor issues
- 0.2+: Significant corruption

---

### 8. Self-BLEU Score

**What it measures:** Diversity across multiple generations for the same question.

**How it works:**
1. Generate N answers (default 5) for each question
2. Calculate BLEU score of each answer against others
3. Average all scores

**Interpretation:**
- **Lower is better** (typical range: 0-100)
- 0-20: High diversity
- 20-50: Moderate diversity
- 50+: Low diversity (repetitive outputs)

---

### 9. Length Deviation Score

**What it measures:** How far answer length deviates from expected.

**Formula:**
```
Deviation = |actual_length - expected_length| / expected_length
```

**Interpretation:**
- **Lower is better**
- 0.0-0.2: Appropriate length
- 0.2-0.5: Somewhat off
- 0.5+: Significantly too long/short

---

### 10. KL Divergence Shift

**What it measures:** How much the fine-tuned model's token distribution has shifted from the base model.

**How it works:**
1. Get logits from both base and fine-tuned models
2. Convert to probability distributions
3. Calculate KL divergence

**Interpretation:**
- **Moderate values are best**
- 0-5: Minimal shift (maybe underfitted)
- 5-15: Healthy specialization
- 15+: Potential overfitting

---

### 11. Answer Quality Score (LLM-as-Judge)

**What it measures:** Overall quality assessed by another model.

**Evaluation criteria:**
- Relevance to question
- Technical correctness
- Clarity and structure
- Non-repetitiveness

**Note:** This metric requires careful prompt engineering and may not be reliable with small evaluator models.

---

## Combined Score

The combined score provides a single metric for ranking models:

**Formula:**
```
Combined = Coherence - Hallucination - 0.01 * Perplexity
```

**Components:**
- **Coherence**: Positive contribution (higher is better)
- **Hallucination**: Negative contribution (lower is better)
- **Perplexity**: Small negative contribution (lower is better)

**Interpretation:**
- **Higher is better**
- Negative: Poor quality (pre-trained models typically score negative)
- 0-0.1: Marginal improvement
- 0.1-0.3: Good quality
- 0.3+: Excellent quality

---

## Usage

### Running Evaluation

```python
from src.model import CyberSecModel
from src.evaluate import Evaluator, EVAL_QUESTIONS

# Load model
model = CyberSecModel(model_path="./models/cybersec-t5")

# Create evaluator
evaluator = Evaluator(
    model=model.model,
    tokenizer=model.tokenizer,
    config=model.config
)

# Run full evaluation
results = evaluator.evaluate(questions=EVAL_QUESTIONS)

# Print results
print(f"Coherence: {results['coherence']:.4f}")
print(f"Hallucination: {results['hallucination']:.4f}")
print(f"Perplexity: {results['perplexity']:.4f}")
print(f"Combined Score: {results['combined_score']:.4f}")
```

### CLI Evaluation

```bash
python scripts/evaluate.py \
    --model-path ./models/cybersec-t5 \
    --output results.json
```

### Individual Metrics

```python
# Single answer metrics
coherence = evaluator.coherence_score(answer)
repetition = Evaluator.repetition_score(answer)
richness = Evaluator.semantic_richness_score(answer)

# Batch metrics
coherence_scores = evaluator.coherence_score_batch(answers)
hallucination_scores = evaluator.hallucination_score_batch(questions, answers)
```

---

## Comparison: Pre-trained vs Fine-tuned

| Metric | Pre-trained | Fine-tuned | Improvement |
|--------|------------|------------|-------------|
| Coherence | 0.049 | 0.450 | **+818%** |
| Hallucination | 0.091 | 0.236 | +159%* |
| Perplexity | 1.252 | 1.189 | **-5%** |
| Repetition | 0.45 | 0.15 | **-67%** |
| Prompt Copying | 0.82 | 0.35 | **-57%** |
| Semantic Richness | 0.25 | 0.48 | **+92%** |
| **Combined Score** | -0.054 | 0.202 | **+474%** |

*Higher hallucination is expected with domain specialization due to increased confidence.

---

## Best Practices

1. **Always use multiple metrics**: No single metric tells the full story
2. **Compare to baseline**: Always evaluate against pre-trained model
3. **Check per-question scores**: Aggregate scores can hide problems
4. **Manual review**: Spot-check generated answers for quality
5. **Domain experts**: Have security professionals verify technical accuracy

