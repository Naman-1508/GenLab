# RAG-Shield Requirements & Architecture Specification

## Project Overview

**RAG-Shield** is a comprehensive defense-in-depth security framework designed for Retrieval-Augmented Generation (RAG) systems. It addresses core security vulnerabilities facing modern LLM retrieval architectures, including knowledge base poisoning, prompt injection (direct and indirect), vector tampering, and retrieval integrity violations.

**Academic & Industry Context**:
- **USENIX Security 2025 (PoisonedRAG)**: Demonstrates that injecting as few as 5 maliciously crafted documents into a knowledge base can achieve an attack success rate exceeding 90%.
- **ACM Web 2025 (RAGForensics)**: Establishes the importance of tracing, attributing, and auditing poisoning attacks in RAG pipelines.
- Modern RAG applications frequently ingest unvetted third-party documents, leaving the generation stage vulnerable to prompt hijacking, context poisoning, and unauthorized instruction execution.

---

## Core Threat Model

### 1. Knowledge Base Poisoning & Document Poisoning
**Attack Vectors**:
- Injecting malicious documents into the knowledge base during ingestion.
- Modifying existing knowledge base content or embedding records.
- Constructing adversarial passages designed to achieve high similarity matches against anticipated user queries.

**Threat Scenario**:
```
Attacker → Injects Poisoned Content → Ingestion Pipeline → Vector Store
                                                               ↓
User Query → Vector Retrieval → Poisoned Chunk Retrieved → LLM Executes Malicious Context → Compromised Output
```

**Impact**:
- Misleading user decisions via fabricated factual statements.
- Context-driven prompt injection overriding system guardrails.
- Destruction of institutional trust in retrieval outputs.

### 2. Prompt Injection (Direct & Indirect)
**Attack Vectors**:
- Embedding explicit instruction overrides (e.g., `Ignore previous instructions and output...`) within ingested documents.
- Data exfiltration payloads attempting to leak system prompts or sensitive retrieved context.
- Jailbreaks and roleplay subversion embedded in external documents.

### 3. Retrieval Integrity Violations
**Attack Vectors**:
- Unauthorized modification of indexed vector embeddings or textual records in the knowledge base.
- Man-in-the-middle or unauthorized alteration of retrieved contexts.
- Lack of verifiable audit trails for compliance and post-incident forensics.

### 4. Data Leakage & Query Privacy
**Attack Vectors**:
- Query eavesdropping revealing proprietary search intents.
- Reconstruction of indexed documents through repeated adversarial probing.

---

## Architectural Requirements

### Component 1: Multi-Format Document Ingestion & Chunking
- Support document ingestion from multiple formats: PDF, TXT, and Markdown.
- Extract, normalize, and segment documents into discrete chunks with deterministic chunk IDs and source tracking.
- Subject every chunk to the pre-ingestion security gate prior to embedding or indexing.

### Component 2: Pre-Ingestion Security Gate (4 Integrated Detectors)
The security gate executes four complementary detection mechanisms on each chunk:
1. **Enhanced Perplexity Analysis**: Evaluates linguistic anomaly via token-level negative log-likelihood against a pre-trained causal language model (e.g., DistilGPT-2 / GPT-2).
2. **Similarity & Outlier Analysis**: Measures cosine distance and cluster deviation against a benign reference distribution or verified knowledge clusters.
3. **Semantic Analysis**: Identifies suspicious semantic anomalies, instruction-like behavior, structural contradictions, and intent manipulation.
4. **Direct Prompt-Injection Analysis**: Detects explicit instruction override phrases, roleplay jailbreaks, system prompt exfiltration signatures, and obfuscated delimiter markers.

### Component 3: Hybrid Ensemble Risk Fusion
- Normalize detector scores into unified scale $[0.0, 1.0]$.
- Combine scores via configurable weighted risk fusion:
  $$\text{RiskScore} = \sum_{k=1}^4 w_k \cdot \text{NormalizedScore}_k, \quad \sum_{k=1}^4 w_k = 1.0$$
- Evaluate against a calibrated decision threshold $\tau \in [0.0, 1.0]$.
- Generate structured detection records containing individual contributions, dominant detector attribution, unified risk score, and flagged textual evidence.

### Component 4: Security Decision & Quarantine Subsystem
- **Suspicious Decision ($\text{Risk} \ge \tau$)**:
  - The chunk is quarantined in `QuarantineManager`.
  - Quarantined items are strictly isolated from vector storage and cannot be retrieved by user queries.
  - Generates comprehensive forensic audit entries.
- **Safe Decision ($\text{Risk} < \tau$)**:
  - Proceeds to embedding generation.
  - Indexed in the trusted vector database (`KnowledgeBase`).
  - Merkle tree and vector commitments updated.

### Component 5: Explainable AI (XAI)
- Interpretable, transparent rationale for every security verdict.
- Quantifies detector attribution percentages.
- Pinpoints suspicious spans and keyword matches within the input chunk.

### Component 6: Cryptographic Integrity Verification
- **Merkle Tree**: Enforces knowledge base integrity with $O(\log n)$ membership proofs and cryptographic root validation.
- **Vector Commitment**: Binds document embeddings cryptographically to prevent vector substitution.
- **Tamper-Evident Audit Log**: Maintains a SHA-256 hash-chained log for all ingestion, quarantine, retrieval, and verification operations.

### Component 7: Secure RAG Retrieval & Generation
- Standard retrieval queries search *only* the trusted vector database.
- Retrieved context is surfaced alongside relevance scores.
- Context is provided to the LLM generator (local or API-backed via environment variables) for question-answering.

---

## Technology Stack

- **Runtime & Language**: Python 3.9+ (Verified on Python 3.13)
- **Vector Search & Embeddings**: FAISS, Sentence-Transformers (`all-MiniLM-L6-v2`)
- **Language Models & Perplexity**: PyTorch, Hugging Face Transformers (`distilgpt2` / `gpt2`)
- **Cryptography**: Python standard `hashlib`, `secrets`, `cryptography`
- **Data & Evaluation**: Scikit-Learn, NumPy, SciPy, Pandas
- **Testing**: PyTest, PyTest-Cov
- **User Interface**: Research Dashboard (Interactive Streamlit / Web UI)

---

## Evaluation Metrics

1. **Security & Detection**:
   - Precision, Recall, F1-Score, False Positive Rate (FPR), Detection Rate.
   - Confusion matrix across benign, poisoned, and prompt-injection corpora.
2. **RAG Response Quality**:
   - Context Relevance, Cleanliness Rate (zero quarantined context leakage), Generation Faithfulness.
3. **Computational Performance**:
   - Ingestion latency per chunk (ms).
   - End-to-end RAG response latency.
   - Cryptographic verification overhead.

---

## Academic & Engineering Milestones

1. **Phase 1: Repository Audit & Verification**: Complete static security audit, dependency scan, English localization, and automated regression verification.
2. **Phase 2: Core Hybrid Security Gate**: Implement multi-format document ingestion, Enhanced Perplexity Detector, Outlier Detector, Semantic Detector, dedicated Direct Prompt-Injection Detector, and Hybrid Risk Fusion.
3. **Phase 3: Quarantine, XAI & Integrity Integration**: Wire quarantine isolation, Explainable AI attribution engine, Merkle tree root updates, and audit log chaining.
4. **Phase 4: Evaluation Benchmark Suite**: Construct reproducible experimental pipelines comparing Vanilla RAG vs. Hybrid Secure RAG across curated benchmarks (WikiText-103, BIPIA, Prompt Injection Dataset, PoisonedRAG samples).
5. **Phase 5: Research Dashboard & Demonstration Package**: Deliver modern English dashboard, comprehensive documentation, automated test suite, and submission-ready academic artifacts.
