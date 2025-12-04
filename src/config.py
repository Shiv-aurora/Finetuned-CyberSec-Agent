"""
Configuration and hyperparameters for CyberSecLLM.

This module contains all configurable parameters for training,
evaluation, and inference.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class Config:
    """Configuration class for CyberSecLLM training and inference."""
    
    # Model Configuration
    model_name: str = "google-t5/t5-small"
    max_input_length: int = 384
    max_output_length: int = 128
    
    # Training Hyperparameters (Best found via grid search)
    learning_rate: float = 3e-5
    batch_size: int = 8
    epochs: int = 3
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    
    # Generation Parameters
    num_beams: int = 1
    do_sample: bool = True
    top_p: float = 0.9
    temperature: float = 0.7
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.1
    
    # Dataset
    dataset_name: str = "Mohabahmed03/Alpaca_Dataset_CyberSecurity_2.0"
    train_samples: Optional[int] = None  # None = use all samples
    filter_prefix: str = "Please provide detailed information about"
    
    # Paths
    output_dir: str = "./outputs"
    model_save_dir: str = "./models"
    
    # Device
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    fp16: bool = field(default_factory=torch.cuda.is_available)
    
    # Evaluation
    eval_batch_size: int = 16
    num_eval_questions: int = 53


# Default evaluation questions for benchmarking
EVAL_QUESTIONS: List[str] = [
    "Explain the difference between symmetric and asymmetric encryption.",
    "What is a firewall and how does it block traffic?",
    "Describe how SQL injection works and how to prevent it.",
    "Explain how encapsulation in the OSI model can be abused during a penetration test.",
    "Give examples of how attackers target specific OSI layers such as Layer 2 or Layer 7.",
    "Why is the Presentation layer relevant for encryption and security operations?",
    "How does the Data Link layer become a target for ARP spoofing attacks?",
    "What role does layer separation play in security hardening efforts?",
    "Describe how improper handling of HTTP request headers can expose sensitive data.",
    "Explain why HTTP status codes may reveal implementation weaknesses during testing.",
    "What dangers arise from poorly sanitized HTTP response bodies?",
    "How can attackers leverage malformed HTTP request lines?",
    "What are common security implications of misconfigured cookies in HTTP headers?",
    "Describe how recursive DNS servers can be abused in cache-poisoning attacks.",
    "Explain how attackers exploit DNS tunneling for data exfiltration.",
    "What characteristics of DGA-based domains allow them to evade detection?",
    "Why is sinkholing an effective defense against DNS malware C2 channels?",
    "How can misconfigured authoritative DNS servers lead to subdomain takeover?",
    "Explain how Kerberoasting works and why service accounts are vulnerable.",
    "What makes the KRBTGT account one of the highest-value targets in an Active Directory domain?",
    "How does unconstrained delegation lead to privilege escalation risks?",
    "Describe how forged TGTs enable Golden Ticket attacks.",
    "How does time synchronization manipulation weaken Kerberos authentication?",
    "Explain how Ajax can accidentally hide malicious payload delivery.",
    "Why does asynchronous content loading complicate security monitoring?",
    "Describe how partial page updates can be leveraged in XSS attacks.",
    "How does Ajax affect the reliability of traditional WAF-based detection?",
    "Why does Ajax increase the need for strict input validation?",
    "How do ATT&CK TTPs help analysts map multi-stage intrusion patterns?",
    "Why is mapping detections to ATT&CK crucial for blue teams?",
    "How can threat intelligence support early detection of common web attacks like SQLi?",
    "Describe how honeypot data collection aligns with ATT&CK persistence techniques.",
    "How can threat hunting be guided by known attacker tradecraft patterns?",
    "Explain the differences between union-based and boolean-based SQL injection.",
    "How does error-based SQL injection help attackers enumerate database structure?",
    "Why do WAFs often fail to catch obfuscated SQL payloads?",
    "Describe how attackers escalate from SQL injection to remote OS command execution.",
    "Explain why database fingerprinting is required for effective SQL injection exploitation.",
    "Describe how XSS bypasses the same-origin policy in modern browsers.",
    "Why are event-handler based payloads often successful in bypassing WAFs?",
    "List ways to exploit DOM-based XSS without sending malicious input to the server.",
    "Explain how cookies and session tokens can be stolen via JavaScript injection.",
    "How can inconsistent browser parsing behaviors enable XSS exploitation?",
    "How is SSRF used to scan internal networks behind firewalls?",
    "Why are cloud metadata services prime SSRF targets?",
    "Describe how SSRF enables attackers to pivot toward additional vulnerabilities like XXE.",
    "What makes SSRF dangerous when paired with permissive URL parsing?",
    "How can SSRF cause denial-of-service via forced internal requests?",
    "Explain why early CGI applications were highly vulnerable to command injection.",
    "How could poor CGI error messages leak sensitive server information?",
    "Why did early Internet systems fail to prevent simple exploitation attempts?",
    "Describe common serverless FaaS attack surfaces introduced by event triggers.",
    "How can misconfigured OAuth redirect URIs result in token theft?",
]


# Prompt template for cybersecurity Q&A
PROMPT_TEMPLATE = "Answer the following cybersecurity question.\n\nQuestion: {question}\nAnswer:"


def format_prompt(question: str) -> str:
    """Format a question using the cybersecurity prompt template."""
    return PROMPT_TEMPLATE.format(question=question)

