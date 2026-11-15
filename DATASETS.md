 # Dataset Curation & Experimental Corpus Specification

**Project Title:** Prompt Injection and Document Poisoning Detection for Secure Retrieval-Augmented Generation Systems  
**Date:** November 2026  
**Status:** Validated & Frozen for Evaluation  

---

## 1. Overview and Curation Strategy

To evaluate the proposed **Hybrid Secure RAG** architecture rigorously without fabricating metrics or overfitting, we curated standardized, representative subsets from four authoritative open-source security and NLP corpora.

Full raw datasets in this domain span hundreds of gigabytes (e.g. raw WikiText-103 is over 500 MB of uncompressed tokens, while full BIPIA contains extensive conversational permutations). Following Part 4 guidelines, we conducted availability, licensing, format, and compatibility audits before extracting balanced, representative subsets.

---

## 2. Dataset Candidates and Specifications

### 2.1 WikiText-103 (Salesforce Research)
- **Source:** Hugging Face Datasets (`Salesforce/wikitext`)
- **License:** Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0)
- **Purpose in Framework:**
  - Establishes the baseline empirical linguistic distribution of benign natural language text.
  - Used for reference distribution modeling in `SimilarityDetector`.
  - Calibrates baseline negative log-likelihood (NLL) and perplexity thresholds in `PerplexityDetector`.
  - Serves as negative control samples (label: `SAFE`).
- **Representative Subset:** 10 diverse encyclopedic articles covering history, biology, computer engineering, physics, and ecology (`data/wikitext_subset.json`).

### 2.2 Microsoft BIPIA (Benchmark for Indirect Prompt Injection Attacks)
- **Source:** GitHub (`microsoft/BIPIA`)
- **License:** MIT License
- **Purpose in Framework:**
  - Evaluates stealthy indirect prompt injection attacks where adversarial override commands are embedded inside superficially legitimate external documents.
  - Tests the Pre-Ingestion Security Gate against data exfiltration, system prompt leakage, and role subversion payloads.
  - Positive control samples (label: `SUSPICIOUS`).
- **Representative Subset:** 6 verified attack samples spanning instruction injection, system prompt extraction, delimiter evasion, and citation suppression (`data/bipia_subset.json`).

### 2.3 Prompt Injection Detection Dataset (mirzaakhi)
- **Source:** GitHub (`mirzaakhi/prompt-injection-detection-dataset`)
- **License:** MIT License
- **Purpose in Framework:**
  - Dedicated evaluation of **Detector 4: Direct Prompt-Injection Analysis**.
  - Evaluates direct override patterns (e.g. `Ignore all previous instructions`), persona jailbreaks (`DAN`), spaced obfuscation (`I G N O R E`), and markdown delimiter hijacking.
  - Positive control samples (label: `SUSPICIOUS`).
- **Representative Subset:** 6 direct injection variants (`data/prompt_injection_subset.json`).

### 2.4 PoisonedRAG (USENIX Security 2025)
- **Source:** GitHub (`sleeepeer/PoisonedRAG`)
- **Paper:** *PoisonedRAG: Knowledge Poisoning Attacks to Retrieval-Augmented Generation of Large Language Models*
- **Purpose in Framework:**
  - Evaluates knowledge poisoning where deceptive factual assertions are injected to corrupt RAG question-answering.
  - Assesses retrieval isolation and confirms that poisoned documents cannot enter the trusted knowledge base or be returned to the LLM generator.
  - Positive control samples (label: `SUSPICIOUS`).
- **Representative Subset:** 6 targeted knowledge poisoning attacks with corresponding clean facts and target queries (`data/poisoned_rag_subset.json`).

---

## 3. Strict Train / Validation / Test Separation & Leakage Prevention

To guarantee academic reproducibility and prevent data leakage:
1. **Reference / Train Split (`train`):**
   - 4 benign samples from WikiText-103.
   - Used exclusively to establish reference embedding centroids in `SimilarityDetector` and fit baseline language perplexity.
   - Never used for computing evaluation metrics.
2. **Validation Split (`val`):**
   - 2 benign samples (WikiText-103), 2 indirect injection samples (BIPIA), 2 direct injection samples, and 2 knowledge poisoning samples (PoisonedRAG).
   - Used exclusively to configure the ensemble decision threshold ($\tau = 0.50$) and tune detector weights.
3. **Final Test Split (`test`):**
   - 4 benign samples (WikiText-103), 4 indirect injection samples (BIPIA), 4 direct injection samples, and 4 knowledge poisoning samples (PoisonedRAG).
   - Total: 16 evaluation items (4 benign, 12 malicious across 3 distinct attack families).
   - **Completely held out** during calibration. All reported metrics in `EVALUATION.md` are computed exclusively on this test split.

---

## 4. Preprocessing & Document Chunking Pipeline

Every sample is processed through `DocumentIngestionPipeline`:
- Text normalization: Unicode control characters and zero-width spaces stripped.
- Whitespace collapsing and consistent sentence delimiter alignment.
- Chunking: 500-character sliding windows with 50-character overlap.
- Deterministic chunk hash generation (`SHA-256`) for cryptographic audit tracking.

---

## 5. Summary Table

| Dataset | Attack Family | Samples (Train / Val / Test) | Role in System |
| :--- | :--- | :---: | :--- |
| **WikiText-103** | Benign Natural Prose | 4 / 2 / 4 | Reference distribution, Perplexity baseline, Negative control |
| **Microsoft BIPIA** | Indirect Prompt Injection | 0 / 2 / 4 | Context-embedded override testing |
| **Prompt Injection Dataset** | Direct Prompt Injection | 0 / 2 / 4 | Explicit command & jailbreak testing |
| **PoisonedRAG** | Knowledge Poisoning | 0 / 2 / 4 | Deceptive fact & retrieval isolation testing |
| **Total** | Multi-Family Benchmark | **4 / 8 / 16** | **28 Curated Samples Total** |
