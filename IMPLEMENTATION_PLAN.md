# Implementation Plan: Hybrid Secure RAG Framework

**Project Title:** Prompt Injection and Document Poisoning Detection for Secure Retrieval-Augmented Generation Systems  
**Target Completion / Submission Date:** 16 November 2026  
**Lead Engineer:** Senior AI/ML, Cybersecurity & RAG Architect  

---

## 1. Project Overview & Architectural Alignment

Conventional Retrieval-Augmented Generation (RAG) systems implicitly trust ingested documents. If malicious documents containing adversarial prompt injections or poisoned facts are indexed into the knowledge base, subsequent user queries can retrieve these chunks, directly subverting LLM generation.

This project transforms the existing `RAG-Shield` repository into an academically rigorous, fully tested, and defensible **Hybrid Secure RAG** system featuring a **Pre-Ingestion Security Gate** with:
1. **Enhanced Perplexity Analysis** (Statistical language modeling via pre-trained causal LM)
2. **Similarity / Outlier Analysis** (Embedding manifold anomaly & reference cluster deviation)
3. **Semantic Analysis** (Instructional semantics, contradiction, and behavioral subversion patterns)
4. **Direct Prompt-Injection Analysis** (Explicit instruction override, roleplay jailbreaks, system prompt exfiltration signatures)
5. **Hybrid Ensemble Risk Fusion** (Normalized weighted fusion & calibrated decision thresholding)
6. **Physical Quarantine & Safe Vector Storage Isolation**
7. **Explainable AI (XAI)** (Attribution of detector contributions and suspicious span extraction)
8. **Cryptographic Integrity Layer** (Merkle Tree & Vector Commitment)
9. **Secure RAG Retrieval & Generation** (Retrieval guarantees strictly over verified documents)
10. **Vanilla RAG vs. Hybrid Secure RAG Comparative Benchmark**

---

## 2. Architecture Specification

```
                          [ External Documents: PDF / TXT / MD ]
                                           │
                                           ▼
                                [ Document Processing ]
                                [ Text Extraction & Normalization ]
                                [ Chunking Engine ]
                                           │
                                    Document Chunks
                                           │
       ┌───────────────────────────────────┴───────────────────────────────────┐
       │                 PRE-INGESTION SECURITY GATE (4 DETECTORS)             │
       │                                                                       │
       │  ┌─────────────────────────┐           ┌───────────────────────────┐  │
       │  │  1. Enhanced Perplexity │           │  2. Similarity / Outlier  │  │
       │  │     (Causal LM: GPT-2)  │           │     (Embeddings / Manifold│  │
       │  └────────────┬────────────┘           └─────────────┬─────────────┘  │
       │               │                                      │                │
       │               │       ┌──────────────────────┐       │                │
       │               ├───────┤ 3. Semantic Analysis ├───────┤                │
       │               │       └──────────┬───────────┘       │                │
       │               │                  │                   │                │
       │  ┌────────────┴────────────┐     │                   │                │
       │  │ 4. Direct Prompt-Inject │     │                   │                │
       │  │    (Rules + Classifier) │     │                   │                │
       │  └────────────┬────────────┘     │                   │                │
       │               │                  │                   │                │
       │               ▼                  ▼                   ▼                │
       │              [ HYBRID ENSEMBLE RISK FUSION MECHANISM ]                │
       │                Normalized Score Aggregation & Weights                 │
       │               Unified Risk Score R in [0.0, 1.0] vs Tau               │
       └───────────────────────────────────┬───────────────────────────────────┘
                                           │
                                [ Security Decision ]
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          [ Risk >= Threshold: SUSPICIOUS ]             [ Risk < Threshold: SAFE ]
                    │                                             │
                    ▼                                             ▼
           [ Quarantine Storage ]                       [ Embedding Generation ]
           (Isolated from Vector DB)                              │
                    │                                             ▼
                    │                                   [ Trusted Vector DB ]
                    │                                   (FAISS / KnowledgeBase)
                    │                                             │
                    │                                             ▼
                    │                                  [ Cryptographic Integrity ]
                    │                                  (Merkle Tree & Vector Commit)
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                            [ Audit Log & XAI Engine ]
                                           │
                                           ▼
                                [ Secure RAG Retriever ]
                                (Queries ONLY Trusted DB)
                                           │
                                           ▼
                                 [ LLM Generator & UI ]
```

---

## 3. Work Breakdown Structure (Phases & Status)

### Phase 1: Repository & Security Audit (Completed)
- [x] Inspect existing directory structure, dependencies, and test suite.
- [x] Execute Bandit security scanner on Python AST.
- [x] Execute pip-audit on installed packages.
- [x] Scan for hardcoded credentials, secret keys, and dangerous invocations.
- [x] Identify and translate legacy Chinese documentation (`REQUIREMENTS.md`).
- [x] Author `SECURITY_AUDIT.md` and `LANGUAGE_AUDIT.md`.

### Phase 2: Core Hybrid Security Gate Implementation
- [ ] **Document Ingestion Module** (`src/ragshield/ingestion/`):
  - Support multi-format inputs: PDF, TXT, and Markdown.
  - Implement deterministic chunking, metadata propagation, and chunk-level tracking.
- [ ] **Detector 1: Enhanced Perplexity Detector** (`src/ragshield/detection/perplexity.py`):
  - Implement token-level negative log-likelihood calculation using pre-trained causal LM (`distilgpt2` or `gpt2`).
  - Implement sliding-window chunk evaluation and threshold calibration.
- [ ] **Detector 2: Similarity & Outlier Detector** (`src/ragshield/detection/similarity.py`):
  - Reference corpus embedding distribution modeling.
  - Distance-to-centroid and nearest-neighbor outlier scoring.
- [ ] **Detector 3: Semantic Detector** (`src/ragshield/detection/semantic.py`):
  - Syntactic instruction structure recognition, semantic conflict scoring, and jailbreak pattern detection.
- [ ] **Detector 4: Direct Prompt-Injection Detector** (`src/ragshield/detection/prompt_injection.py`):
  - Dedicated 4th detector for direct injection attempts (system overrides, prompt leakage, delimiter attacks).
  - Rule-based regex baseline + feature-based classification.
- [ ] **Hybrid Ensemble Risk Fusion** (`src/ragshield/detection/fusion.py`):
  - Score normalization, weighted linear combination, and calibrated threshold decision.
  - Generation of structured `FusionResult` with chunk attribution.

### Phase 3: Quarantine, XAI, Integrity & Secure RAG
- [ ] **Quarantine Subsystem Integration**:
  - Route suspicious chunks to physical quarantine.
  - Cryptographically ensure quarantined chunks cannot be indexed into the FAISS/KnowledgeBase index.
- [ ] **Explainable AI (XAI) Module** (`src/ragshield/xai/explainer.py`):
  - Quantitative feature attribution (% contribution per detector).
  - Suspicious text span and pattern extraction for human inspection.
- [ ] **Integrity Layer Verification**:
  - Merkle Tree root calculation on safe ingestion.
  - Vector commitment generation and verification.
  - SHA-256 hash-chained verifiable audit logging.
- [ ] **Secure RAG Pipeline** (`src/ragshield/core/rag_system.py`):
  - Verified retriever querying exclusively safe vector storage.
  - Generation pipeline with retrieved context transparency.

### Phase 4: Datasets & Comparative Evaluation
- [ ] **Dataset Preparation** (`src/ragshield/benchmarks/` & `data/`):
  - Prepare curated benign text (WikiText-103 subset).
  - Prepare prompt injection attacks (BIPIA subset + Prompt Injection Dataset).
  - Prepare document poisoning samples (PoisonedRAG subset).
- [ ] **Evaluation Harness**:
  - Compare **System A: Vanilla RAG** vs. **System B: Proposed Hybrid Secure RAG**.
  - Compute security metrics: Precision, Recall, F1, FPR, Detection Rate.
  - Compute RAG quality & integrity metrics: Context Cleanliness, Retrieval Faithfulness.
  - Compute latency overhead per chunk and per query.
  - Conduct ablation studies (Perplexity only, Similarity only, Semantic only, Injection only, Full Ensemble).
  - Produce real CSV tables and visual evaluation plots.

### Phase 5: Modern Research Dashboard & Final Artifacts
- [ ] **English Research Dashboard**:
  - Multi-tab UI for Document Ingestion & Chunk Security Inspection, Knowledge Base & Quarantine Browser, Secure RAG Question-Answering, and Live Evaluation Benchmarks.
- [ ] **Documentation**:
  - `README.md`, `ARCHITECTURE.md`, `METHODOLOGY.md`, `DATASETS.md`, `EVALUATION.md`, `.env.example`.
- [ ] **Automated Test Suite**:
  - Unit tests covering all 4 detectors, fusion, quarantine, XAI, integrity, and RAG retrieval.
- [ ] **Demonstration Guide**:
  - Clear, reproducible walk-through instructions for the university presentation on 16 November 2026.

---

## 4. Verification & Defense Criteria

Every component must be backed by real code, genuine outputs, passing automated tests, and reproducible benchmarks. No fabricated scores or mocked results will be included.

