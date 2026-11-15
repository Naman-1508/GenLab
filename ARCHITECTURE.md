# System Architecture: Hybrid Secure RAG

**Project Title:** Prompt Injection and Document Poisoning Detection for Secure Retrieval-Augmented Generation Systems  
**Academic Target Date:** 16 November 2026  
**Security Framework:** RAG-Shield 0.2.0 (Defense-in-Depth)  

---

## 1. High-Level Architectural Overview

Conventional Retrieval-Augmented Generation (RAG) pipelines operate under an **implicit trust assumption**: they assume all documents added to the knowledge base are benign. Consequently, when adversaries inject prompt override instructions (Direct or Indirect Prompt Injection) or fabricated factual statements (Knowledge Poisoning), these malicious passages are indexed, retrieved, and incorporated into the LLM context, inducing context hijacking, privacy leakage, or hallucinated outputs.

**Hybrid Secure RAG** introduces a **Pre-Ingestion Security Gate** positioned between external document processing and vector database storage. Every document chunk is independently evaluated by four orthogonal detection mechanisms, fused into a Unified Risk Score, and routed through strict physical isolation. Quarantined content can **never** enter the trusted knowledge base or be returned by the retriever.

```
+---------------------------------------------------------------------------------------------------+
|                                      EXTERNAL UNTRUSTED DOCUMENTS                                 |
|                                       (PDF / TXT / Markdown Files)                                |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|                                  DOCUMENT INGESTION PIPELINE                                      |
|       - Text Extraction (pypdf / raw stream)                                                      |
|       - Normalization (Unicode control stripping, whitespace collapse, line ending cleanup)      |
|       - Overlapping Chunking Engine (Sliding window: chunk_size=500, overlap=50, deterministic ID)|
+---------------------------------------------------------------------------------------------------+
                                                  │
                                            Document Chunks
                                                  │
                                                  ▼
+───────────────────────────────────────────────────────────────────────────────────────────────────+
│                                 PRE-INGESTION SECURITY GATE                                       │
│                                                                                                   │
│  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌───────────────────────────────────┐  │
│  │   Detector 1: Enhanced  │  │   Detector 2:           │  │   Detector 3:                     │  │
│  │   Perplexity Analysis   │  │   Similarity / Outlier  │  │   Semantic Analysis               │  │
│  │  - Causal LM (DistilGPT2│  │  - Manifold Clusters    │  │  - Instruction structure analysis │  │
│  │  - Token-level NLL Loss │  │  - Cosine Distance      │  │  - Factual contradiction markers  │  │
│  │  - Normalized Anomaly s │  │  - Outlier score s_sim  │  │  - Intent override score s_sem    │  │
│  └────────────┬────────────┘  └────────────┬────────────┘  └─────────────────┬─────────────────┘  │
│               │                            │                                 │                    │
│               │               ┌────────────┴────────────────────────┐        │                    │
│               │               │   Detector 4: Direct Prompt-Inject  │        │                    │
│               │               │  - Rule-based regex baseline        │        │                    │
│               │               │  - Delimiter & canary detection     │        │                    │
│               │               │  - Normalized injection score s_inj │        │                    │
│               │               └────────────┬────────────────────────┘        │                    │
│               │                            │                                 │                    │
│               ▼                            ▼                                 ▼                    │
│  +─────────────────────────────────────────────────────────────────────────────────────────────+  │
│  │                         HYBRID ENSEMBLE RISK FUSION MECHANISM                               │  │
│  │                         R = w_ppl*s_ppl + w_sim*s_sim + w_sem*s_sem + w_inj*s_inj           │  │
│  │                         Calibrated Threshold Evaluation: R >= tau ?                         │  │
│  +─────────────────────────────────────────┬───────────────────────────────────────────────────+  │
+────────────────────────────────────────────┼──────────────────────────────────────────────────────+
                                             │
                                   [ Security Decision ]
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
          [ R >= tau: SUSPICIOUS ]                       [ R < tau: SAFE ]
                      │                                             │
                      ▼                                             ▼
             [ Quarantine Storage ]                    [ Embedding Generation ]
           - Physically isolated from DB               - Sentence-Transformers
           - Cannot be searched or leaked                           │
                      │                                             ▼
                      │                                   [ Trusted Vector DB ]
                      │                                   - FAISS Cosine Index
                      │                                             │
                      │                                             ▼
                      │                                  [ Cryptographic Integrity ]
                      │                                  - Merkle Tree Root Update
                      │                                  - Vector Commitments
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             │
                                             ▼
                              [ Verifiable Audit Log (SHA-256) ]
                              [ Explainable AI (XAI) Attribution ]
                                             │
                                             ▼
                              [ Secure RAG Retrieval Engine ]
                              (Queries strictly Trusted Vector DB)
                                             │
                                      Verified Context
                                             │
                                             ▼
                                  [ LLM Answer Generator ]
```

---

## 2. Module-Wise Detailed Specifications

### Module 1: Document Ingestion Pipeline (`src/ragshield/ingestion/processor.py`)
- **Input:** External files or data streams (`.pdf`, `.txt`, `.md`, raw strings).
- **Processing:**
  1. *Extraction:* Uses `pypdf` with byte-stream fallback for PDF text extraction; standard UTF-8 stream decoder for text/markdown.
  2. *Normalization:* Strips invisible unicode (zero-width spaces `\u200b-\u200f`, directionality overrides), normalizes line endings, and collapses repeated whitespace.
  3. *Chunking:* Implements configurable windowing (`chunk_size=500` characters, `overlap=50` characters) with sentence boundary look-back to avoid splitting thoughts mid-sentence.
  4. *Metadata Packaging:* Emits discrete `Document` instances with `doc_id=parent_chunk_index`, cryptographic content SHA-256 hash, and provenance metadata.
- **Output:** List of `Document` chunks ready for independent security analysis.

---

### Module 2: Pre-Ingestion Security Gate (4 Integrated Detectors)

#### Detector 1: Enhanced Perplexity Analysis (`src/ragshield/detection/perplexity.py`)
- **Input:** Document chunk text.
- **Processing:**
  - Evaluates token-level negative log-likelihood (NLL) against a pre-trained causal language model (DistilGPT-2 or GPT-2).
  - Sliding-window evaluation with stride $S=256$ handles arbitrarily long chunks.
  - Perplexity equation:
    $$\text{PPL}(X) = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log P(x_i \mid x_{<i}) \right)$$
  - Converts raw perplexity to normalized risk score $s_{\text{ppl}} \in [0.0, 1.0]$ via a calibrated logistic function centered at threshold $\theta_{\text{ppl}}$.
- **Output:** Normalized score $s_{\text{ppl}}$, raw loss, token count, threat classification.

#### Detector 2: Similarity & Outlier Analysis (`src/ragshield/detection/similarity.py`)
- **Input:** Document chunk and vector embedding $\mathbf{e}$.
- **Processing:**
  - Compares the chunk's normalized embedding vector against cluster centroids $\{\mathbf{c}_j\}_{j=1}^K$ derived from a trusted benign reference corpus (WikiText-103).
  - Computes cosine outlier distance:
    $$d_{\text{outlier}} = 1.0 - \max_{j} \cos(\mathbf{e}, \mathbf{c}_j)$$
  - Evaluates coordinate structural properties: vector $L_2$ norm, coordinate entropy, and sparsity ratio to detect synthetic adversarial vectors.
- **Output:** Normalized outlier score $s_{\text{sim}} \in [0.0, 1.0]$, centroid distance, and distribution anomaly flags.

#### Detector 3: Semantic Analysis (`src/ragshield/detection/semantic.py`)
- **Input:** Document chunk text.
- **Processing:**
  - Inspects semantic and structural intent for imperative instruction commands embedded in factual passages.
  - Evaluates contradiction markers and factual subversion patterns (e.g. `the true and official answer is`, `everyone is wrong`, `disregard all other documents`).
  - Detects covert RAG override directives (`note to assistant:`, `when the user asks about X, say Y`).
- **Output:** Normalized semantic risk score $s_{\text{sem}} \in [0.0, 1.0]$, matched contradiction tags, and evidence spans.

#### Detector 4: Direct Prompt-Injection Analysis (`src/ragshield/detection/prompt_injection.py`)
- **Input:** Document chunk text.
- **Processing:**
  - Executes comprehensive high-precision deterministic rules targeting direct override commands (`ignore previous instructions`, `disregard safety guidelines`).
  - Identifies system prompt exfiltration directives (`reveal your system prompt`, `print hidden instructions`).
  - Identifies persona jailbreaks (`DAN`, `unrestricted assistant`, `FreedAI`).
  - Detects raw LLM delimiter hijacking (`<<SYS>>`, `[INST]`, ````system`).
  - Identifies obfuscated spaced-letter evasion (`I G N O R E  P R E V I O U S`).
- **Output:** Normalized injection score $s_{\text{inj}} \in [0.0, 1.0]$, matched rule identifiers, and exact flagged text spans.

---

### Module 3: Hybrid Ensemble Risk Fusion (`src/ragshield/detection/fusion.py`)
- **Input:** Scores $(s_{\text{ppl}}, s_{\text{sim}}, s_{\text{sem}}, s_{\text{inj}})$ and fusion weights $(w_1, w_2, w_3, w_4)$.
- **Processing:**
  - Computes the Unified Risk Score:
    $$R = \sum_{k=1}^4 w_k \cdot s_k, \quad \sum_{k=1}^4 w_k = 1.0$$
    Default weights: $w_{\text{ppl}}=0.20, w_{\text{sim}}=0.25, w_{\text{sem}}=0.25, w_{\text{inj}}=0.30$.
  - Applies calibrated decision threshold $\tau = 0.50$.
  - Includes a critical override: high-confidence direct injection ($s_{\text{inj}} \ge 0.80$) or critical semantic subversion ($s_{\text{sem}} \ge 0.50$) elevates $R$ and forces a suspicious classification.
  - Computes percentage attribution per detector:
    $$\text{Contrib}_k = \frac{w_k \cdot s_k}{R} \times 100\%$$
- **Output:** `UnifiedSecurityResult` containing chunk verdict (`safe` vs `suspicious`), $R$, dominant detector, and flagged evidence.

---

### Module 4: Quarantine Subsystem & Security Decision (`src/ragshield/defense/quarantine.py`)
- **If Suspicious ($R \ge \tau$):**
  - Chunk is stored in `QuarantineManager`.
  - Cryptographically isolated: the chunk is **never** passed to the embedder for indexing into the trusted FAISS vector store.
  - Generates forensic event record in `AuditLog`.
- **If Safe ($R < \tau$):**
  - Chunk is embedded using Sentence-Transformers (`all-MiniLM-L6-v2`).
  - Vector commitment generated.
  - Added to `KnowledgeBase`.
  - Merkle tree leaf hash updated.

---

### Module 5: Explainable AI (XAI) (`src/ragshield/xai/explainer.py`)
- **Input:** `UnifiedSecurityResult` and chunk content.
- **Processing:**
  - Calculates threshold safety margin $\Delta = R - \tau$.
  - Quantifies feature attribution across all 4 detectors.
  - Translates quantitative metrics into a concise human-readable narrative explanation.
  - Renders visual Markdown explanation audit cards.
- **Output:** `ExplanationReport` structured object and markdown card.

---

### Module 6: Cryptographic Integrity Layer (`src/ragshield/integrity/`)
- **Merkle Tree:**
  - Binary hash tree using SHA-256 with domain separation prefixes (`00:` for leaves, `01:` for internal nodes).
  - Provides $O(\log n)$ membership proofs.
  - Any unauthorized modification to indexed document text alters the root hash.
- **Vector Commitment:**
  - Cryptographic binding for floating-point embedding coordinates via Merkle commitment scheme.
  - Prevents covert embedding perturbation or vector substitution attacks.
- **Audit Log:**
  - Tamper-evident append-only log with SHA-256 hash chaining:
    $$H_i = \text{SHA256}(i \parallel \text{timestamp} \parallel \text{op\_type} \parallel \text{data} \parallel H_{i-1})$$
  - Any modification, deletion, or re-ordering breaks the hash chain.

---

### Module 7: Secure RAG Retrieval & Generation (`src/ragshield/core/secure_rag.py`)
- **Query Flow:**
  $$\text{Query} \to \text{Query Embedding} \to \text{FAISS Retriever} \to \text{Verified Contexts} \to \text{LLM Generator} \to \text{Answer}$$
- **Retrieval Isolation Guarantee:**
  - The retriever searches *strictly* within the trusted vector database.
  - The pipeline programmatically audits retrieved IDs against the quarantine registry, verifying 0% leakage.
- **Generation:** Contextually grounds answers in verified contexts, transparently citing source document IDs and relevance scores.

