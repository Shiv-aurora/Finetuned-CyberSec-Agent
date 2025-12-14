"""
CyberSecLLM Gradio Demo Application

A Hugging Face Spaces-compatible demo for the CyberSecLLM model.
Provides an interactive interface for asking cybersecurity questions.

To run locally:
    pip install gradio transformers torch
    python app/app.py
    
For Hugging Face Spaces, this file serves as the main entry point.
"""

import gradio as gr
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ============================================================
# Configuration
# ============================================================

# Model configuration - update this to your HF model ID after upload
MODEL_ID = "shiv-aurora/cybersec-t5-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sample questions for the demo
SAMPLE_QUESTIONS = [
    "What is SQL injection and how can it be prevented?",
    "Explain the difference between symmetric and asymmetric encryption.",
    "How does a firewall protect a network?",
    "What is cross-site scripting (XSS)?",
    "Explain how Kerberos authentication works.",
    "What is a DDoS attack and how can organizations defend against it?",
    "Describe how SSRF attacks work.",
    "What are the main layers of the OSI model and their security implications?",
]

# ============================================================
# Model Loading
# ============================================================

print(f"Loading model: {MODEL_ID}")
print(f"Device: {DEVICE}")

tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")


# ============================================================
# Inference Function
# ============================================================

def format_prompt(question: str) -> str:
    """Format the question using the cybersecurity prompt template."""
    return f"Answer the following cybersecurity question.\n\nQuestion: {question}\nAnswer:"


def generate_answer(
    question: str,
    max_length: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate an answer to a cybersecurity question.
    
    Args:
        question: The cybersecurity question to answer
        max_length: Maximum length of generated answer
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling threshold
        
    Returns:
        Generated answer string
    """
    if not question.strip():
        return "Please enter a cybersecurity question."
    
    # Format the prompt
    prompt = format_prompt(question)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=384,
    ).to(DEVICE)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_length,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            no_repeat_ngram_size=3,
            repetition_penalty=1.1,
            early_stopping=True,
        )
    
    # Decode
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return answer


# ============================================================
# Gradio Interface
# ============================================================

# Custom CSS for cybersecurity theme
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.main-title {
    text-align: center;
    color: #00d4aa;
    margin-bottom: 0.5em;
}

.subtitle {
    text-align: center;
    color: #6c757d;
    margin-bottom: 2em;
}

.example-btn {
    border: 1px solid #00d4aa !important;
}

footer {
    display: none !important;
}
"""

# Create the interface
with gr.Blocks(css=CUSTOM_CSS, title="CyberSecLLM Demo") as demo:
    
    # Header
    gr.HTML("""
        <h1 class="main-title">🛡️ CyberSecLLM</h1>
        <p class="subtitle">Fine-tuned T5 for Cybersecurity Question Answering</p>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input section
            question_input = gr.Textbox(
                label="Ask a Cybersecurity Question",
                placeholder="e.g., What is SQL injection and how can it be prevented?",
                lines=3,
            )
            
            with gr.Row():
                submit_btn = gr.Button("🔍 Get Answer", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")
            
            # Output section
            answer_output = gr.Textbox(
                label="Answer",
                lines=8,
                interactive=False,
            )
        
        with gr.Column(scale=1):
            # Settings
            gr.Markdown("### ⚙️ Generation Settings")
            
            max_length_slider = gr.Slider(
                minimum=32,
                maximum=256,
                value=128,
                step=16,
                label="Max Length",
            )
            
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.1,
                label="Temperature",
            )
            
            top_p_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="Top-p (Nucleus Sampling)",
            )
    
    # Sample questions
    gr.Markdown("### 💡 Try These Examples")
    with gr.Row():
        for i in range(0, 4):
            gr.Button(
                SAMPLE_QUESTIONS[i][:50] + "...",
                size="sm",
            ).click(
                fn=lambda q=SAMPLE_QUESTIONS[i]: q,
                outputs=question_input,
            )
    
    with gr.Row():
        for i in range(4, 8):
            gr.Button(
                SAMPLE_QUESTIONS[i][:50] + "...",
                size="sm",
            ).click(
                fn=lambda q=SAMPLE_QUESTIONS[i]: q,
                outputs=question_input,
            )
    
    # Event handlers
    submit_btn.click(
        fn=generate_answer,
        inputs=[question_input, max_length_slider, temperature_slider, top_p_slider],
        outputs=answer_output,
    )
    
    question_input.submit(
        fn=generate_answer,
        inputs=[question_input, max_length_slider, temperature_slider, top_p_slider],
        outputs=answer_output,
    )
    
    clear_btn.click(
        fn=lambda: ("", ""),
        outputs=[question_input, answer_output],
    )
    
    # Footer info
    gr.Markdown("""
    ---
    ### About
    
    **CyberSecLLM** is a T5-small model fine-tuned on 159,000+ cybersecurity Q&A pairs.
    
    - 📈 **818% improvement** in answer coherence vs base T5
    - ⚡ **73% smaller** with INT8 quantization
    - 🎯 Covers SQLi, XSS, encryption, Kerberos, and more
    
    [📚 GitHub](https://github.com/Shiv-aurora/Finetuned-CyberSec-Agent) | 
    [🤗 Model Hub](https://huggingface.co/shiv-aurora/cybersec-t5-small) |
    [📖 Documentation](https://github.com/Shiv-aurora/Finetuned-CyberSec-Agent/tree/main/docs)
    
    ---
    *Note: This model is for educational purposes. Always verify security information with trusted sources.*
    """)


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
    )

