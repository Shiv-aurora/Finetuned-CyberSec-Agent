# Sample Model Outputs

This document showcases example outputs from both the pre-trained T5-small model and our fine-tuned CyberSecLLM, demonstrating the significant improvement in domain-specific knowledge.

## Example 1: SQL Injection

**Question:** *Describe how SQL injection works and how to prevent it.*

### Pre-trained T5 Output
> Answer the following cybersecurity question: Describe how SQL injection works and how to prevent it. Answer: Describe how SQL injection works and how to prevent it.

❌ **Issue:** Simply echoes the question without providing any actual information.

### Fine-tuned CyberSecLLM Output
> The SQL injection process involves a combination of a SQL injection and a sanitization of the database. This can be done by modifying the database to a specific type of SQL injection, allowing the attacker to execute arbitrary SQL queries, and then executing arbitrary scripts. Then, the attacker can execute a malicious SQL query, which is a powerful SQL injection tool that can be used to execute SQL queries.

✅ **Improvement:** Provides actual cybersecurity knowledge about SQL injection mechanisms.

---

## Example 2: OSI Model Attacks

**Question:** *Explain how encapsulation in the OSI model can be abused during a penetration test.*

### Pre-trained T5 Output
> Answer the following cybersecurity question: Explain how encapsulation in the OSI model can be abused during a penetration test. Answer:

❌ **Issue:** No meaningful response generated.

### Fine-tuned CyberSecLLM Output
> During a penetration test, encapsulation in the OSI model can be abused by attackers. This could lead to unauthorized access to sensitive information, such as passwords, user privileges, and data breaches. Additionally, attackers can exploit this vulnerability to gain access to vulnerable systems, potentially leading to unpatched attacks.

✅ **Improvement:** Explains security implications and attack vectors.

---

## Example 3: Encryption Types

**Question:** *Explain the difference between symmetric and asymmetric encryption.*

### Pre-trained T5 Output
> Answer the following cybersecurity question: Explain the difference between symmetric and asymmetric encryption. Answer: Explain the difference between symmetric and asymmetric encryption.

❌ **Issue:** Complete failure to generate useful content.

### Fine-tuned CyberSecLLM Output
> The difference between symmetric and asymmetric encryption is that of asymmetric encryption. Symmetric encryption uses a single key for both encryption and decryption, while asymmetric encryption uses a pair of keys. For example, if asymmetric encryption is asymmetric, the attackers can use the symmetric encryption to access the symmetric key.

✅ **Improvement:** Demonstrates understanding of cryptographic concepts.

---

## Metrics Summary

| Metric | Pre-trained | Fine-tuned | Change |
|--------|------------|------------|--------|
| Coherence | 0.049 | 0.450 | **+818%** |
| Perplexity | 1.252 | 1.189 | **-5%** |
| Combined Score | -0.054 | 0.202 | **+474%** |

---

## Evaluation Criteria

Our evaluation metrics assess multiple dimensions of answer quality:

1. **Coherence** - Logical flow between sentences (using sentence embeddings)
2. **Hallucination** - Consistency and confidence in responses
3. **Perplexity** - Language modeling fluency
4. **Repetition** - Avoidance of repeated phrases
5. **Semantic Richness** - Vocabulary diversity

See [METRICS.md](../docs/METRICS.md) for detailed metric explanations.

