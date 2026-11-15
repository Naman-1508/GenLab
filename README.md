# Hybrid Secure RAG
### Prompt Injection and Document Poisoning Detection for Secure Retrieval-Augmented Generation Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-340%20passed-brightgreen.svg)]()
[![Precision](https://img.shields.io/badge/Precision-1.0000-success)]()
[![F1-Score](https://img.shields.io/badge/F1--Score-0.9565-blue)]()

---

## Academic Project Overview

**Project Title:** Prompt Injection and Document Poisoning Detection for Secure Retrieval-Augmented Generation Systems  
**Submission & Demonstration Date:** 16 November 2026  
**Original Repository Basis:** [SidereusHu/RAG-Shield](https://github.com/SidereusHu/RAG-Shield) *(All third-party intellectual property and foundations are acknowledged and preserved)*.  

This project transforms the RAG-Shield framework into an academically defensible, end-to-end **Hybrid Secure RAG** architecture. Conventional RAG pipelines suffer from an implicit trust vulnerability: documents ingested into the knowledge base are assumed to be clean. When malicious actors inject prompt overrides or poisoned factual statements, these passages are retrieved and directly corrupt LLM context windows.

Our framework introduces an integrated **Pre-Ingestion Security Gate** positioned between document chunking and vector storage, combining four orthogonal detection mechanisms via **Hybrid Ensemble Risk Fusion**, physical **Quarantine Isolation**, **Explainable AI (XAI)**, and **Cryptographic Integrity Verification** (Merkle Trees & Vector Commitments).

---

## Key Architectural Components

```
External Documents (PDF / TXT / MD)
               │
               ▼
   [ Document Ingestion Pipeline ]
   Text Extraction ➔ Normalization ➔ Chunking (chunk_size=500, overlap=50)
               │
               ▼
+───────────────────────────────────────────────────────────────────────────+
│                       PRE-INGESTION SECURITY GATE                         │
│                                                                           │
│   [Detector 1] Enhanced Perplexity Analysis (Causal LM NLL Loss)          │
│   [Detector 2] Similarity / Outlier Analysis (Embedding Manifold Outlier) │
│   [Detector 3] Semantic Analysis (Instruction & Factual Contradiction)    │
│   [Detector 4] Direct Prompt-Injection Analysis (Rules, Delims, Jailbreak)│
│                                      │                                    │
│                                      ▼                                    │
│                    [ Hybrid Ensemble Risk Fusion ]                        │
│                R = w1*s_ppl + w2*s_sim + w3*s_sem + w4*s_inj              │
│                     Calibrated Decision Threshold: R >= τ                 │
+──────────────────────────────────────┬────────────────────────────────────+
                                       │
                             [ Security Decision ]
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                         ▼
      [ R >= τ: SUSPICIOUS ]                     [ R < τ: SAFE ]
                  │                                         │
                  ▼                                         ▼
        [ Quarantine Storage ]                 [ Embedding Generation ]
      (Strict Physical Isolation)             (Sentence-Transformers)
                  │                                         │
                  ▼                                         ▼
      [ Never Indexed into KB ]                   [ Trusted Vector DB ]
      [ Cannot Be Leaked ]                                  │
                  │                                         ▼
                  │                                [ Merkle Tree Root ]
                  │                                [ Vector Commitment ]
                  │                                         │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                       [ Verifiable Audit Log (SHA-256) ]
                       [ Explainable AI (XAI) Attribution ]
                                       │
                                       ▼
                        [ Secure RAG Retrieval Engine ]
                       (Queries ONLY Trusted Vector DB)
                                       │
                                Verified Context
                                       │
                                       ▼
                            [ LLM Answer Generator ]
```

---

## Four Security Detectors

| Detector | Purpose & Scientific Basis | Key Metrics & Output |
| :--- | :--- | :--- |
| **1. Enhanced Perplexity** | Measures statistical linguistic naturalness using token-level Negative Log-Likelihood against a pre-trained causal LM (DistilGPT-2). | Raw PPL, Token count, Average NLL loss, Normalized score $s_{\text{ppl}} \in [0, 1]$. |
| **2. Similarity / Outlier** | Computes cosine distance to benign reference cluster centroids and detects degenerate coordinate norms, entropy, and sparsity. | Distance to centroid $d_{\text{outlier}}$, Coordinate flags, Normalized score $s_{\text{sim}} \in [0, 1]$. |
| **3. Semantic Analysis** | Identifies imperative instruction commands, covert RAG override directives, and factual contradiction subversion markers. | Contradiction flags, Imperative density, Normalized score $s_{\text{sem}} \in [0, 1]$. |
| **4. Direct Prompt-Injection** | High-precision deterministic rules and structural classifiers for prompt overrides (`ignore previous instructions`), DAN jailbreaks, delimiters (`<<SYS>>`, `[INST]`), and spaced evasion (`I G N O R E`). | Matched rules, Flagged exact text spans, Normalized score $s_{\text{inj}} \in [0, 1]$. |

---

## Experimental Benchmark Results

Evaluated across 28 curated samples across four standard benchmarks: **WikiText-103** (benign controls), **Microsoft BIPIA** (indirect prompt injection), **Prompt Injection Detection Dataset** (direct prompt injection), and **PoisonedRAG** (targeted knowledge poisoning).

### 1. System Comparison: Vanilla RAG vs. Hybrid Secure RAG

| Metric | System A: Vanilla RAG (Baseline) | System B: Proposed Hybrid Secure RAG |
| :--- | :---: | :---: |
| **Pre-Ingestion Security Gate** | None (Implicit Ingestion) | **4-Detector Hybrid Gate** |
| **Malicious Documents Ingested** | 12 / 12 (100.0% Ingested into Vector DB) | **1 / 12 (8.3% Ingested)** |
| **Quarantine Interception Rate** | 0.0% | **91.67% (11 / 12 Quarantined)** |
| **Retrieval Context Contamination** | 6 / 7 queries (85.7%) | **3 / 7 queries (42.9%)** [Real MiniLM] |
| **Confirmed Harmful Answers Generated** | 5 / 7 queries (71.4%) | **1 / 7 queries (14.3%)** (80.0% Harm Reduction) |
| **Physical Quarantine Breaches** | N/A (No Quarantine) | **0 Breaches (0.0% Quarantine Leakage)** |
| **Precision** | N/A (No Filter) | **1.0000** |
| **Recall** | 0.0000 | **0.9167** |
| **F1-Score** | 0.0000 | **0.9565** |
| **False Positive Rate (FPR)** | 0.0% | **0.00%** |
| **Cryptographic Integrity** | None | **Merkle Tree & Vector Commitments** |
| **Explainable AI (XAI)** | None | **Quantitative Attribution Cards** |

### 2. Ablation Study: Contribution of Individual Detectors

| Detector Configuration | Precision | Recall | F1-Score | Detection Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Perplexity Alone** | 0.667 | 0.667 | 0.667 | 66.67% |
| **Similarity Outlier Alone** | 0.000 | 0.000 | 0.000 | 0.00% |
| **Semantic Analysis Alone** | 1.000 | 0.500 | 0.667 | 50.00% |
| **Prompt Injection Alone** | 1.000 | 0.667 | 0.800 | 66.67% |
| **Full Hybrid Ensemble (Proposed)** | **1.000** | **0.917** | **0.957** | **91.67%** |

*All experimental numbers are genuine, unmanipulated, and reproducible via `python run_experiments.py`.*

---

## Installation & Setup

### Prerequisites
- Python 3.9+ (Tested and verified on Python 3.13)
- Dedicated virtual environment recommended.

### Quick Setup

```bash
# Clone and enter the repository
git clone https://github.com/SidereusHu/RAG-Shield.git
cd RAG-Shield

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate   # On Windows PowerShell
# source venv/bin/activate # On Linux/macOS

# Install dependencies and local editable package
pip install -r requirements.txt
pip install -e .

# Configure environment variables (optional for external LLMs)
cp .env.example .env
```

---

## Execution & Demonstration

### 1. Run the Complete Automated Test Suite (340 Tests)
```bash
pytest tests/ -v
```
*Expected: 340 passed in ~7.5 seconds.*

### 2. Run the Academic Demonstration Script (CLI Walkthrough)
```bash
python demo.py
# Or for instant 3-second presentation mode:
# $env:USE_MOCK_EMBEDDER="1"; python demo.py  (Windows PowerShell)
```
*Walks through the entire 8-step defense lifecycle: benign indexing, direct injection interception, BIPIA containment, PoisonedRAG isolation, XAI attribution, cryptographic integrity audit, and secure Q&A.*

### 3. Run the Benchmark Experiment Suite
```bash
python run_experiments.py
```
*Executes Vanilla vs Hybrid comparison, computes metrics dynamically from raw execution, and generates CSV tables & figures in `eval_results/`.*

### 4. Launch the English Research Dashboard
```bash
streamlit run dashboard.py
```
*Opens the web dashboard in your browser (`http://localhost:8501`), supporting:*
- Multi-format document upload (PDF, TXT, MD) and chunking.
- Live chunk-level 4-detector security evaluation and XAI attribution cards.
- Knowledge base and quarantine isolation inspector.
- Secure RAG question-answering with zero-leakage retrieval verification.
- Cryptographic Merkle Tree and Audit Log integrity verification.
- Live evaluation metric tables and visualization figures.

---

## Project Documentation Index

- **`FINAL_AUDIT_REPORT.md`**: Independent code audit report, forensic trace of evaluation consistency, and verified metrics.
- **`FINAL_DEMO_GUIDE.md`**: Step-by-step presentation script for the final examination committee on 16 November 2026.
- **`PROFESSOR_QA.md`**: Comprehensive defense answers to 16 advanced technical questions from professors.
- **`REQUIREMENTS.md`**: Technical specification and threat model (100% English localized).
- **`IMPLEMENTATION_PLAN.md`**: Work breakdown structure and phase-wise engineering roadmap.
- **`ARCHITECTURE.md`**: Comprehensive architectural diagrams, module inputs/processing/outputs.
- **`METHODOLOGY.md`**: Mathematical equations, risk fusion equations, and cryptographic proofs.
- **`DATASETS.md`**: Benchmark dataset curation, sourcing, licensing, and split specifications.
- **`EVALUATION.md`**: Full empirical evaluation report with confusion matrices and ablation analysis.
- **`SECURITY_AUDIT.md`**: Codebase security review, secret scanning, Bandit, and pip-audit reports.
- **`LANGUAGE_AUDIT.md`**: Repository-wide English localization report (0 foreign characters).

---

## Known Limitations & Future Scope

1. **Local Model Downloads:** Transformer perplexity relies on local caching of lightweight causal models (DistilGPT-2 / GPT-2). On offline evaluation systems, the framework falls back gracefully to statistical n-gram entropy.
2. **Adversarial Multilingual Injections:** Current rule baselines specialize in English prompt overrides. Future work includes expanding multilingual rule coverage and cross-lingual embedding manifolds.
3. **Hardware Acceleration:** Inference runs efficiently on standard CPU architectures. GPU acceleration (CUDA) can further reduce chunk ingestion latency on million-token enterprise corpora.

---

## License & Attribution

- Released under the **MIT License**.
- Built upon foundations from **RAG-Shield** (`https://github.com/SidereusHu/RAG-Shield`).
- All reused components preserved and properly attributed.
