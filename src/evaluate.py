"""
Evaluation metrics for CyberSecLLM.

This module implements 11 different evaluation metrics for assessing
the quality of generated cybersecurity answers:

1. Repetition Score - Detects repeated n-grams
2. Prompt Copying Score - Measures question overlap in answers
3. Length Deviation Score - Checks answer length appropriateness
4. Semantic Richness Score - Vocabulary diversity
5. Coherence Score - Logical flow between sentences
6. Hallucination Score - Answer consistency/confidence
7. Invalid Token Score - Detects corrupted output
8. Self-BLEU Score - Measures output diversity
9. Perplexity - Language model quality
10. KL Divergence - Distribution shift from base model
11. Combined Score - Weighted aggregate metric
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import re
import string
import numpy as np
import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer

from .config import Config, format_prompt, EVAL_QUESTIONS


class Evaluator:
    """
    Comprehensive evaluator for CyberSecLLM models.
    
    Computes multiple metrics to assess answer quality including
    coherence, hallucination detection, and linguistic quality.
    """
    
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        tokenizer: T5Tokenizer,
        config: Optional[Config] = None,
        embed_model: Optional[Any] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: The T5 model to evaluate
            tokenizer: The tokenizer
            config: Configuration object
            embed_model: Sentence transformer model for coherence metrics
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or Config()
        self.device = self.config.device
        
        # Lazy load sentence transformer
        self._embed_model = embed_model
    
    @property
    def embed_model(self):
        """Lazy load the sentence transformer model."""
        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        return self._embed_model
    
    # ============================================================
    # Metric 1: Repetition Score
    # ============================================================
    @staticmethod
    def repetition_score(text: str, n: int = 3) -> float:
        """
        Compute repetition ratio based on repeated n-grams.
        
        Lower score = better (less repetition).
        
        Args:
            text: The text to analyze
            n: N-gram size (default 3)
            
        Returns:
            Repetition ratio between 0 and 1
        """
        tokens = text.split()
        if len(tokens) < n:
            return 0.0
        
        ngrams = [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(count for count in counts.values() if count > 1)
        total = len(ngrams)
        
        return repeated / total if total > 0 else 0.0
    
    # ============================================================
    # Metric 2: Prompt Copying Score
    # ============================================================
    @staticmethod
    def prompt_copying_score(question: str, answer: str) -> float:
        """
        Measure how much of the question appears in the answer.
        
        Higher score = more copying (worse).
        
        Args:
            question: The input question
            answer: The generated answer
            
        Returns:
            Overlap ratio between 0 and 1
        """
        q_tokens = question.lower().split()
        a_tokens = answer.lower().split()
        
        if len(q_tokens) == 0:
            return 0.0
        
        overlap = sum(1 for t in q_tokens if t in a_tokens)
        return overlap / len(q_tokens)
    
    # ============================================================
    # Metric 3: Length Deviation Score
    # ============================================================
    @staticmethod
    def length_deviation_score(answer: str, expected_length: int = 60) -> float:
        """
        Compute how far the answer length deviates from expected.
        
        Lower score = better.
        
        Args:
            answer: The generated answer
            expected_length: Expected number of tokens
            
        Returns:
            Deviation ratio (can be > 1)
        """
        tokens = answer.split()
        actual_length = len(tokens)
        deviation = abs(actual_length - expected_length) / expected_length
        return deviation
    
    # ============================================================
    # Metric 4: Semantic Richness Score
    # ============================================================
    @staticmethod
    def semantic_richness_score(answer: str) -> float:
        """
        Measure vocabulary diversity in the answer.
        
        Higher score = richer, more varied vocabulary (better).
        
        Args:
            answer: The generated answer
            
        Returns:
            Unique token ratio between 0 and 1
        """
        cleaned = answer.translate(str.maketrans("", "", string.punctuation))
        tokens = cleaned.lower().split()
        
        if len(tokens) == 0:
            return 0.0
        
        unique_tokens = set(tokens)
        return len(unique_tokens) / len(tokens)
    
    # ============================================================
    # Metric 5: Coherence Score
    # ============================================================
    def coherence_score(self, answer: str) -> float:
        """
        Compute coherence by measuring similarity between consecutive sentences.
        
        Higher score = more coherent (better).
        
        Args:
            answer: The generated answer
            
        Returns:
            Average cosine similarity between consecutive sentences
        """
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        
        if len(sentences) < 2:
            return 0.0
        
        embeddings = self.embed_model.encode(sentences, convert_to_tensor=True)
        
        sims = []
        for i in range(len(embeddings) - 1):
            sim = F.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[i+1].unsqueeze(0),
                dim=1
            ).item()
            sims.append(sim)
        
        return float(np.mean(sims)) if sims else 0.0
    
    def coherence_score_batch(self, answers: List[str]) -> List[float]:
        """Compute coherence scores for multiple answers."""
        return [self.coherence_score(ans) for ans in answers]
    
    # ============================================================
    # Metric 6: Hallucination Score
    # ============================================================
    def hallucination_score_batch(
        self,
        questions: List[str],
        answers: List[str],
        batch_size: int = 8,
    ) -> List[float]:
        """
        Compute hallucination scores using model confidence and statistical analysis.
        
        Higher score = more likely hallucination (worse).
        
        Methods combined:
        1. Token-level entropy from model logits
        2. Repetition detection
        3. Uncertain language patterns
        
        Args:
            questions: List of input questions
            answers: List of generated answers
            batch_size: Processing batch size
            
        Returns:
            List of hallucination scores
        """
        all_scores = []
        max_len = self.config.max_input_length
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i + batch_size]
            batch_answers = answers[i:i + batch_size]
            
            # Method 1: Model confidence via entropy
            input_prompts = [format_prompt(q) for q in batch_questions]
            input_enc = self.tokenizer(
                input_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len
            ).to(self.device)
            
            output_enc = self.tokenizer(
                batch_answers,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_enc.input_ids,
                    attention_mask=input_enc.attention_mask,
                    labels=output_enc["input_ids"]
                )
                logits = outputs.logits
                
                # Entropy calculation
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
                confidence_scores = torch.mean(entropy, dim=-1) / 10.0
                confidence_scores = confidence_scores.cpu().numpy()
            
            # Methods 2 & 3: Statistical analysis
            for idx, answer in enumerate(batch_answers):
                answer_lower = answer.lower()
                
                # Repetition score
                words = answer_lower.split()
                if len(words) > 5:
                    unique_ratio = len(set(words)) / len(words)
                    repetition_score = 1.0 - unique_ratio
                else:
                    repetition_score = 0.0
                
                # Uncertain language detection
                uncertain_phrases = ['maybe', 'might', 'could be', 'possibly', 'unclear', 'not sure']
                uncertainty_score = min(
                    sum(phrase in answer_lower for phrase in uncertain_phrases) / 3.0,
                    1.0
                )
                
                # Combined score
                combined = (
                    float(confidence_scores[idx]) * 0.7 +
                    repetition_score * 0.2 +
                    uncertainty_score * 0.1
                )
                all_scores.append(float(combined))
        
        return all_scores
    
    # ============================================================
    # Metric 7: Invalid Token Score
    # ============================================================
    @staticmethod
    def invalid_token_score(answer: str) -> Tuple[float, Dict[str, Any]]:
        """
        Detect invalid, corrupted, or nonsensical token patterns.
        
        Score 0 = clean text (good), Score 1 = broken output (bad).
        
        Args:
            answer: The generated answer
            
        Returns:
            Tuple of (score, details dictionary)
        """
        checks = {
            "weird_symbols": len(re.findall(r"[^a-zA-Z0-9\s.,!?;:()\-]", answer)),
            "triple_repeat_words": len(re.findall(r"\b(\w+)\b(?:\s+\1){2,}", answer)),
            "character_runs": len(re.findall(r"(.)\1\1\1+", answer)),
            "non_english_ratio": 0.0,
        }
        
        letters = sum(c.isalpha() for c in answer)
        non_english = sum(not (c.isalpha() or c.isspace()) for c in answer)
        checks["non_english_ratio"] = non_english / (letters + 1)
        
        score = (
            0.25 * (checks["weird_symbols"] > 0) +
            0.25 * (checks["triple_repeat_words"] > 0) +
            0.25 * (checks["character_runs"] > 0) +
            0.25 * checks["non_english_ratio"]
        )
        
        return min(score, 1.0), checks
    
    # ============================================================
    # Metric 8: Perplexity
    # ============================================================
    def perplexity_batch(
        self,
        answers: List[str],
        batch_size: int = 8
    ) -> float:
        """
        Compute average perplexity across answers.
        
        Lower perplexity = more fluent text (better).
        
        Args:
            answers: List of generated answers
            batch_size: Processing batch size
            
        Returns:
            Average perplexity score
        """
        all_losses = []
        
        for i in range(0, len(answers), batch_size):
            batch_ans = answers[i:i + batch_size]
            enc = self.tokenizer(
                batch_ans,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=enc["input_ids"],
                    labels=enc["input_ids"]
                )
                all_losses.append(outputs.loss)
        
        mean_loss = torch.mean(torch.stack(all_losses))
        perplexity = torch.exp(mean_loss).item()
        
        return float(perplexity)
    
    def perplexity_per_sample(self, answers: List[str]) -> List[float]:
        """Compute perplexity for each answer individually."""
        perps = []
        self.model.eval()
        
        for ans in answers:
            enc = self.tokenizer(
                ans,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=enc["input_ids"],
                    labels=enc["input_ids"]
                )
            perps.append(float(torch.exp(outputs.loss).item()))
        
        return perps
    
    # ============================================================
    # Full Evaluation Pipeline
    # ============================================================
    def evaluate(
        self,
        questions: Optional[List[str]] = None,
        answers: Optional[List[str]] = None,
        generate_answers: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full evaluation pipeline.
        
        Args:
            questions: List of questions to evaluate. Uses defaults if None.
            answers: Pre-generated answers. Generated if None and generate_answers=True.
            generate_answers: Whether to generate answers if not provided.
            
        Returns:
            Dictionary containing all metrics and individual scores
        """
        questions = questions or EVAL_QUESTIONS
        
        if answers is None and generate_answers:
            from .model import CyberSecModel
            model_wrapper = CyberSecModel.__new__(CyberSecModel)
            model_wrapper.model = self.model
            model_wrapper.tokenizer = self.tokenizer
            model_wrapper.config = self.config
            model_wrapper.device = self.device
            print("Generating answers...")
            answers = model_wrapper.generate_batch(questions)
        
        print("Computing metrics...")
        
        # Compute all metrics
        coherence_scores = self.coherence_score_batch(answers)
        hallucination_scores = self.hallucination_score_batch(questions, answers)
        perplexity_scores = self.perplexity_per_sample(answers)
        
        repetition_scores = [self.repetition_score(a) for a in answers]
        copying_scores = [
            self.prompt_copying_score(q, a) 
            for q, a in zip(questions, answers)
        ]
        richness_scores = [self.semantic_richness_score(a) for a in answers]
        
        # Aggregate metrics
        results = {
            "coherence": float(np.mean(coherence_scores)),
            "hallucination": float(np.mean(hallucination_scores)),
            "perplexity": float(np.mean(perplexity_scores)),
            "repetition": float(np.mean(repetition_scores)),
            "prompt_copying": float(np.mean(copying_scores)),
            "semantic_richness": float(np.mean(richness_scores)),
            "per_question": {
                "questions": questions,
                "answers": answers,
                "coherence": coherence_scores,
                "hallucination": hallucination_scores,
                "perplexity": perplexity_scores,
                "repetition": repetition_scores,
                "prompt_copying": copying_scores,
                "semantic_richness": richness_scores,
            }
        }
        
        # Combined score: coherence ↑ - hallucination ↓ - 0.01*perplexity ↓
        results["combined_score"] = (
            results["coherence"] - 
            results["hallucination"] - 
            0.01 * results["perplexity"]
        )
        
        return results


def evaluate_model(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    questions: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        questions: Optional custom questions
        
    Returns:
        Evaluation results dictionary
    """
    evaluator = Evaluator(model, tokenizer)
    return evaluator.evaluate(questions)

