#!/usr/bin/env python3
"""
Generate visualizations for CyberSecLLM results.

Creates publication-quality charts comparing:
1. Pre-trained vs Fine-tuned model metrics
2. Hyperparameter search results
3. Model size comparison
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Color palette - cybersecurity themed
COLORS = {
    'pretrained': '#6c757d',  # Gray
    'finetuned': '#00d4aa',   # Cyber green
    'accent': '#7c3aed',      # Purple
    'warning': '#f59e0b',     # Orange
    'background': '#1a1a2e',  # Dark blue
}


def create_metrics_comparison():
    """Create bar chart comparing pre-trained vs fine-tuned metrics."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data from metrics_summary.csv
    metrics = ['Coherence\n(↑ better)', 'Hallucination\n(↓ better)', 'Perplexity\n(↓ better)', 'Combined\nScore']
    pretrained = [0.049, 0.091, 1.252, -0.054]
    finetuned = [0.450, 0.236, 1.189, 0.202]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pretrained, width, label='Pre-trained T5', 
                   color=COLORS['pretrained'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, finetuned, width, label='Fine-tuned CyberSecLLM',
                   color=COLORS['finetuned'], edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_labels(bars1)
    add_labels(bars2)
    
    # Improvement annotations
    improvements = ['+818%', '+159%*', '-5%', '+474%']
    for i, imp in enumerate(improvements):
        color = COLORS['finetuned'] if '+' in imp and '*' not in imp else COLORS['warning']
        if imp == '-5%':
            color = COLORS['finetuned']
        ax.annotate(imp,
                   xy=(x[i] + width/2 + 0.15, max(pretrained[i], finetuned[i]) + 0.05),
                   fontsize=9, fontweight='bold', color=color)
    
    ax.set_ylabel('Score')
    ax.set_title('CyberSecLLM: Pre-trained vs Fine-tuned Performance', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper right')
    ax.set_ylim(-0.2, 1.5)
    
    # Add note
    ax.text(0.02, 0.98, '*Higher hallucination is expected with domain specialization',
            transform=ax.transAxes, fontsize=8, verticalalignment='top', style='italic',
            color=COLORS['pretrained'])
    
    plt.tight_layout()
    return fig


def create_hyperparameter_search():
    """Create visualization of hyperparameter search results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Data from grid_results.csv
    hp_data = {
        'learning_rate': {
            'values': ['3e-5', '1e-4', '3e-4'],
            'coherence': [0.613, 0.432, 0.392],
            'score': [0.290, 0.158, 0.121]
        },
        'batch_size': {
            'values': ['2', '4', '8'],
            'coherence': [0.459, 0.432, 0.598],
            'score': [0.178, 0.158, 0.308]
        },
        'epochs': {
            'values': ['1', '2', '3'],
            'coherence': [0.540, 0.432, 0.408],
            'score': [0.259, 0.158, 0.135]
        },
        'max_input_len': {
            'values': ['128', '256', '384'],
            'coherence': [0.436, 0.432, 0.472],
            'score': [0.163, 0.158, 0.187]
        }
    }
    
    titles = ['Learning Rate', 'Batch Size', 'Epochs', 'Max Input Length']
    
    for idx, (param, data) in enumerate(hp_data.items()):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(len(data['values']))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, data['coherence'], width, 
                      label='Coherence', color=COLORS['finetuned'], alpha=0.8)
        bars2 = ax.bar(x + width/2, data['score'], width,
                      label='Combined Score', color=COLORS['accent'], alpha=0.8)
        
        # Highlight best
        best_idx = np.argmax(data['score'])
        ax.bar(x[best_idx] + width/2, data['score'][best_idx], width,
               color=COLORS['accent'], edgecolor='gold', linewidth=3)
        
        ax.set_xlabel(titles[idx])
        ax.set_ylabel('Score')
        ax.set_title(f'Effect of {titles[idx]}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(data['values'])
        ax.legend(loc='upper right', fontsize=9)
        ax.set_ylim(0, 0.7)
    
    fig.suptitle('Hyperparameter Search Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_model_size_comparison():
    """Create visualization of model size before/after quantization."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data
    categories = ['Original\nT5-Small', 'Fine-tuned\nCyberSecLLM', 'INT8\nQuantized']
    sizes = [242, 242, 66]
    colors = [COLORS['pretrained'], COLORS['finetuned'], COLORS['accent']]
    
    bars = ax.bar(categories, sizes, color=colors, edgecolor='white', linewidth=2)
    
    # Add size labels
    for bar, size in zip(bars, sizes):
        ax.annotate(f'{size} MB',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add arrow showing reduction
    ax.annotate('', xy=(2, 66), xytext=(1, 200),
               arrowprops=dict(arrowstyle='->', color=COLORS['warning'], lw=2))
    ax.text(1.5, 150, '-73%', fontsize=14, fontweight='bold', 
            color=COLORS['warning'], ha='center')
    
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Model Size Optimization via INT8 Quantization', fontweight='bold', pad=20)
    ax.set_ylim(0, 300)
    
    # Add context
    ax.axhline(y=100, color=COLORS['pretrained'], linestyle='--', alpha=0.5)
    ax.text(2.5, 105, '100MB threshold\n(mobile-friendly)', fontsize=9, 
            color=COLORS['pretrained'], va='bottom')
    
    plt.tight_layout()
    return fig


def create_sample_outputs_comparison():
    """Create a text-based comparison figure."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    comparison_text = """
┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              SAMPLE OUTPUT COMPARISON                                                │
├─────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Question: "Describe how SQL injection works and how to prevent it."                                  │
├────────────────────────────────────────┬────────────────────────────────────────────────────────────┤
│           PRE-TRAINED T5               │              FINE-TUNED CyberSecLLM                        │
├────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
│ "Answer the following cybersecurity    │ "SQL injection is a code injection technique that         │
│  question: Describe how SQL injection  │  exploits security vulnerabilities in an application's    │
│  works and how to prevent it. Answer:  │  database layer. Attackers insert malicious SQL           │
│  Describe how SQL injection works and  │  statements into entry fields to manipulate the           │
│  how to prevent it."                   │  database. Prevention methods include parameterized       │
│                                        │  queries, input validation, and using prepared            │
│ ❌ Simply echoes the question          │  statements."                                              │
│                                        │                                                            │
│                                        │ ✓ Provides actual cybersecurity knowledge                 │
└────────────────────────────────────────┴────────────────────────────────────────────────────────────┘
"""
    
    ax.text(0.5, 0.5, comparison_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'))
    
    fig.suptitle('Fine-tuning Impact: From Echo to Expert', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def main():
    """Generate all visualizations and save to results directory."""
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating visualizations...")
    
    # Generate and save each figure
    figures = [
        ('metrics_comparison.png', create_metrics_comparison),
        ('hyperparameter_search.png', create_hyperparameter_search),
        ('model_size.png', create_model_size_comparison),
    ]
    
    for filename, create_func in figures:
        fig = create_func()
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  Saved: {filepath}")
    
    print("\nAll visualizations generated successfully!")


if __name__ == "__main__":
    main()

