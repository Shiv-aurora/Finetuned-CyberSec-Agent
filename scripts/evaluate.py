#!/usr/bin/env python3
"""
CLI script for evaluating CyberSecLLM models.

Usage:
    python scripts/evaluate.py --model-path ./models/cybersec-t5
    python scripts/evaluate.py --model-path ./models/cybersec-t5 --output results.json
"""

import argparse
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import Config, EVAL_QUESTIONS
from src.model import CyberSecModel
from src.evaluate import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a CyberSecLLM model"
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the model to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON format)"
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        default=None,
        help="Path to custom questions file (one per line)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("CyberSecLLM Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print("=" * 60)
    
    # Load model
    model_wrapper = CyberSecModel(model_path=args.model_path)
    
    # Load custom questions if provided
    questions = EVAL_QUESTIONS
    if args.questions_file:
        with open(args.questions_file, "r") as f:
            questions = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(questions)} custom questions")
    
    # Run evaluation
    evaluator = Evaluator(
        model=model_wrapper.model,
        tokenizer=model_wrapper.tokenizer,
        config=model_wrapper.config,
    )
    
    results = evaluator.evaluate(questions=questions)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Coherence:        {results['coherence']:.4f}")
    print(f"Hallucination:    {results['hallucination']:.4f}")
    print(f"Perplexity:       {results['perplexity']:.4f}")
    print(f"Repetition:       {results['repetition']:.4f}")
    print(f"Prompt Copying:   {results['prompt_copying']:.4f}")
    print(f"Semantic Richness:{results['semantic_richness']:.4f}")
    print("-" * 60)
    print(f"Combined Score:   {results['combined_score']:.4f}")
    print("=" * 60)
    
    # Save results if output specified
    if args.output:
        # Remove per_question data for cleaner JSON (can be very large)
        output_results = {k: v for k, v in results.items() if k != "per_question"}
        output_results["num_questions"] = len(questions)
        
        with open(args.output, "w") as f:
            json.dump(output_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Print sample outputs
    print("\n" + "=" * 60)
    print("Sample Outputs (first 3)")
    print("=" * 60)
    per_q = results["per_question"]
    for i in range(min(3, len(per_q["questions"]))):
        print(f"\nQ: {per_q['questions'][i]}")
        print(f"A: {per_q['answers'][i][:200]}...")
        print(f"   Coherence: {per_q['coherence'][i]:.3f}, "
              f"Halluc: {per_q['hallucination'][i]:.3f}, "
              f"PPL: {per_q['perplexity'][i]:.3f}")


if __name__ == "__main__":
    main()

