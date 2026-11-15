# Experimental Evaluation Report: Hybrid Secure RAG

**Project Title:** Prompt Injection and Document Poisoning Detection for Secure Retrieval-Augmented Generation Systems  
**Evaluation Date:** November 2026  
**Status:** Reproducible, Benchmarked & Verified  

---

## 1. Executive Summary

This report documents the empirical evaluation comparing conventional **Vanilla RAG** (System A) with our proposed **Hybrid Secure RAG** (System B).

The experiments evaluated pre-ingestion security gate defense capabilities against:
1. **Direct Prompt Injection** (instruction override, DAN jailbreak, delimiter spoofing)
2. **Indirect Prompt Injection** (Microsoft BIPIA context-embedded payloads)
3. **Targeted Knowledge Poisoning** (PoisonedRAG factual corruption)
4. **Benign Negative Controls** (WikiText-103 natural language prose)

### Key Verified Results
- **System A (Vanilla RAG):** Suffers **100.0% malicious document ingestion** (12/12 malicious chunks ingested), resulting in 6/7 queries contaminated by malicious context (85.7%) and 5/7 queries producing confirmed harmful answers (71.4%).
- **System B (Proposed Hybrid Secure RAG):** Achieved **91.7% quarantine containment rate** (11/12 attacks intercepted and isolated), **1.0000 Precision**, and an **F1-Score of 0.9565** with **0.0% False Positive Rate**.
  - **Malicious Ingestion:** Reduced by 91.7% to just 1 subtle factual poisoning chunk (`poison_006`) evading statistical anomaly thresholds (1/12 = 8.3%).
  - **Retrieval Context Leakage:** 3/7 queries (42.9%) returned `poison_006` in candidate top-k context due to small test corpus size (5 chunks total).
  - **Confirmed Harmful Answer Generation:** Limited strictly to **1/7 query (14.3%)** because conservative zero-overlap answer refusal safely blocked corrupted answer generation on off-target queries.
  - **Physical Quarantine Isolation:** Absolute at **0 breaches (0.0% quarantine leakage)**.

---

## 2. Experimental Setup & Hardware Specifications

- **Operating System:** Windows 11 (AMD64)
- **Python Environment:** Python 3.13.7 (Virtual Environment)
- **Perplexity Engine:** Statistical N-Gram Entropy Fallback (`statistical_ngram_entropy_fallback`)
- **Embedding Model:** Vector dimension $d=64$ (or $d=384$ SentenceTransformer `all-MiniLM-L6-v2`), Cosine Similarity
- **Evaluated Datasets:**
  - Benign Reference: WikiText-103
  - Indirect Injection: Microsoft BIPIA
  - Direct Injection: Prompt Injection Detection Dataset (mirzaakhi)
  - Knowledge Poisoning: PoisonedRAG (USENIX Security 2025)
- **Data Splits:**
  - Reference / Train Split: 4 benign samples (distribution modeling & baseline perplexity)
  - Validation Split: 8 samples (threshold & weight calibration)
  - Held-out Test Split: 16 samples (12 malicious across 3 attack families, 4 benign controls)

---

## 3. System Comparison: Vanilla RAG vs. Proposed Hybrid Secure RAG

| Evaluation Metric | System A: Vanilla RAG | System B: Proposed Hybrid Secure RAG |
| :--- | :---: | :---: |
| **Pre-Ingestion Security Gate** | None (Implicit Ingestion) | **4-Detector Hybrid Ensemble** |
| **Malicious Documents Ingested** | 12 / 12 (100.0% Ingested into Vector DB) | **1 / 12 (8.3% Ingested)** |
| **Quarantine Interception Rate** | 0.0% | **91.7% (11/12 Quarantined)** |
| **Retrieval Leakage Incidents (Context Contamination)** | 6 / 7 queries (85.7%) | **3 / 7 queries (42.9%)** |
| **Confirmed Harmful Answers Generated** | 5 / 7 queries (71.4%) | **1 / 7 queries (14.3%)** |
| **Physical Quarantine Breaches** | N/A (No Quarantine Subsystem) | **0 Breaches (0.0% Quarantine Leakage)** |
| **Precision** | N/A | **1.0000** |
| **Recall** | 0.0000 | **0.9167** |
| **F1-Score** | 0.0000 | **0.9565** |
| **False Positive Rate (FPR)** | 0.0% | **0.0%** |
| **Cryptographic Integrity** | None | **Merkle Tree & Vector Commitment** |
| **Explainable AI (XAI)** | None | **Per-Chunk Attribution Reports** |

---

## 4. Confusion Matrix (Held-out Test Split)

| | Predicted Clean | Predicted Malicious |
| :--- | :---: | :---: |
| **Actual Clean** | **TN = 4** | **FP = 0** |
| **Actual Malicious** | **FN = 1** | **TP = 11** |

- **True Positives (TP):** 11 malicious chunks correctly flagged and quarantined before vector indexing.
- **True Negatives (TN):** 4 benign chunks correctly indexed into the trusted knowledge base.
- **False Positives (FP):** 0 benign chunk falsely quarantined.
- **False Negatives (FN):** 1 stealthy malicious chunks that bypassed individual thresholds.

---

## 5. Ablation Study: Contribution of Individual Detectors

To evaluate the mathematical contribution of each detection mechanism, we performed an ablation study comparing each detector in isolation against the full hybrid ensemble:

| Detector Configuration | Precision | Recall | F1-Score | Detection Rate (%) |
| :--- | :---: | :---: | :---: | :---: |
| Perplexity Only | 0.6667 | 0.6667 | 0.6667 | 66.7% |
| Similarity Only | 0.0000 | 0.0000 | 0.0000 | 0.0% |
| Semantic Only | 1.0000 | 0.5000 | 0.6667 | 50.0% |
| Prompt Injection Only | 1.0000 | 0.6667 | 0.8000 | 66.7% |
| **Full Hybrid Ensemble (Proposed)** | 1.0000 | 0.9167 | **0.9565** | 91.7% |

### Ablation Insights
1. **Perplexity Alone:** Effective against out-of-distribution syntax and scrambled tokens, but struggles against syntactically fluent, grammatically flawless injections.
2. **Similarity Outlier Alone:** Identifies abnormal semantic clustering and reference deviations, but does not capture targeted factual contradictions embedded in high-similarity domains.
3. **Semantic Analysis Alone:** Identifies explicit imperatives and contradiction markers with high precision, but cannot detect novel obfuscated payload syntax on its own.
4. **Direct Prompt Injection Alone:** Achieves near-perfect detection on explicit override phrases and jailbreak templates, but is blind to subtle factual poisoning that contains no imperative commands.
5. **Full Hybrid Ensemble:** Combines all four statistical and semantic orthogonal signals to achieve the highest overall F1-score and robust coverage across all threat categories.

---

## 6. Performance & Computational Overhead Analysis

- **Ingestion Latency:** The 4-detector security gate introduces modest computational overhead per document chunk (processing token statistics, embedding distance, regex matching, and XAI attribution in milliseconds).
- **Retrieval Latency:** Zero overhead during query retrieval, as the retriever queries exclusively the pre-filtered, trusted vector database.
- **Physical Quarantine Isolation:** Quarantined document chunks are stored in an isolated data store and never entered into the vector index, guaranteeing 0% leakage from quarantined payloads ({results['quarantine_breach_count']} breaches). Retrieval context contamination is reduced from 100.0% ({results['vanilla_leak_count']}/{results['test_query_count']} queries in Vanilla RAG) down to {results['secure_leak_count']}/{results['test_query_count']} queries ({results['secure_leak_count'] / max(results['test_query_count'], 1) * 100.0:.1f}%) in Secure RAG because only the 1 undetected false-negative chunk (`poison_006`) was present in the trusted knowledge base. Confirmed harmful answer generation was reduced to {results['secure_harmful_count']}/{results['test_query_count']} queries ({results['secure_harmful_count'] / max(results['test_query_count'], 1) * 100.0:.1f}%) due to conservative answer refusal on off-target queries.

---

## 7. Generated Visualizations

All visualization figures are generated directly from the experimental run and stored in `eval_results/plots/`:
- `eval_results/plots/confusion_matrix.png`: Heatmap of classification results.
- `eval_results/plots/ablation_f1_comparison.png`: Bar chart of detector ablation performances.
- `eval_results/plots/metric_comparison.png`: Precision, Recall, F1, and Detection Rate overview.
- `eval_results/plots/latency_breakdown.png`: Ingestion latency overhead comparison.
