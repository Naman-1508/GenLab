# Methodology & Mathematical Formulation: Hybrid Secure RAG

**Project Title:** Prompt Injection and Document Poisoning Detection for Secure Retrieval-Augmented Generation Systems  
**Date:** November 2026  
**Audience:** Academic Reviewers & University Evaluation Committee  

---

## 1. Problem Formulation & Threat Model

Let a conventional RAG system maintain a knowledge base $\mathcal{D} = \{d_1, d_2, \dots, d_M\}$ where each document $d_i$ is mapped to an embedding vector $\mathbf{e}_i = \text{Embed}(d_i) \in \mathbb{R}^D$ and indexed in a vector similarity store. When a user presents query $q$, the retriever searches for the top-$k$ nearest neighbors:

$$\mathcal{C}_q = \text{TopK}_{d_i \in \mathcal{D}} \cos(\text{Embed}(q), \mathbf{e}_i)$$

The LLM generator then synthesizes an answer conditioned on context $\mathcal{C}_q$:

$$\hat{y} = \text{LLM}(q, \mathcal{C}_q)$$

### Adversarial Vector Ingestion
An adversary supplies a poisoned or adversarial document $d^*$. In an unprotected RAG system, $d^*$ is ingested directly:
1. **Direct / Indirect Prompt Injection:** $d^*$ contains imperative tokens $I$ instructing the LLM to ignore system instructions or exfiltrate private data.
2. **Knowledge Poisoning:** $d^*$ contains false facts $F^*$ designed to achieve high cosine similarity with target query $q^*$, replacing truthful answers with malicious assertions.

---

## 2. Multi-Detector Formulation (Pre-Ingestion Security Gate)

The Pre-Ingestion Security Gate partitions an incoming document $d$ into discrete normalized chunks $\{c_1, c_2, \dots, c_m\}$. Each chunk $c$ is evaluated independently across four orthogonal security detectors.

### 2.1 Detector 1: Enhanced Perplexity Analysis (Statistical Anomaly)
Adversarial injections, jailbreak suffixes, and obfuscated tokens often deviate statistically from natural linguistic distributions. Given causal language model parameters $\theta$ (e.g. DistilGPT-2):

$$\log P_\theta(c) = \sum_{t=1}^T \log P_\theta(x_t \mid x_{<t})$$

The raw perplexity is computed as:

$$\text{PPL}(c) = \exp\left( -\frac{1}{T} \sum_{t=1}^T \log P_\theta(x_t \mid x_{<t}) \right)$$

To map arbitrary positive perplexity values into a normalized risk score $s_{\text{ppl}} \in [0.0, 1.0]$, we apply a logistic sigmoid centered at anomaly threshold $\theta_{\text{ppl}}$ with slope parameter $\alpha = 0.05$:

$$s_{\text{ppl}}(c) = \frac{1}{1 + \exp\left(-\alpha (\text{PPL}(c) - \theta_{\text{ppl}})\right)}$$

Threshold $\theta_{\text{ppl}}$ is calibrated against the 95th percentile of empirical perplexities on benign text (WikiText-103).

### 2.2 Detector 2: Similarity / Outlier Analysis (Embedding Manifold Deviation)
Let $\{\mathbf{c}_j\}_{j=1}^K$ represent normalized centroids of a verified benign corpus embedding distribution in $\mathbb{R}^D$. For chunk embedding $\mathbf{e} = \text{Embed}(c)$:

$$d_{\text{outlier}}(\mathbf{e}) = 1.0 - \max_{j \in \{1,\dots,K\}} \frac{\mathbf{e} \cdot \mathbf{c}_j}{\|\mathbf{e}\| \|\mathbf{c}_j\|}$$

In addition, we compute coordinate structural integrity features:
- **Vector Norm Anomaly:** $\delta_{\text{norm}} = \mathbb{I}[\|\mathbf{e}\|_2 < 0.5 \lor \|\mathbf{e}\|_2 > 2.0]$
- **Coordinate Entropy:** $H(\mathbf{e}) = -\sum_{i=1}^D p_i \log p_i$, where $p_i = \frac{|e_i|}{\sum_j |e_j|}$
- **Degenerate Sparsity:** $\rho(\mathbf{e}) = \frac{1}{D} \sum_{i=1}^D \mathbb{I}[|e_i| < 10^{-3}]$

The normalized similarity outlier score $s_{\text{sim}} \in [0.0, 1.0]$ aggregates manifold distance with coordinate consistency:

$$s_{\text{sim}}(c) = 0.70 \min\left(1.0, \frac{d_{\text{outlier}}}{\theta_{\text{outlier}}}\right) + 0.30 \left(\frac{\delta_{\text{norm}} + \mathbb{I}[H < 2.0] + \mathbb{I}[\rho > 0.90]}{3}\right)$$

### 2.3 Detector 3: Semantic Analysis (Factual & Intent Inconsistency)
This detector evaluates semantic structure, contradiction markers, and covert RAG manipulation. Given text $c$, we compute:
1. Pattern match density $M_{\text{pat}}(c) = \min(1.0, 0.40 + 0.30 \cdot |\{\text{patterns matched}\}|)$
2. Imperative instruction keyword density $M_{\text{kw}}(c) = \min(1.0, 0.20 + 0.15 \cdot |\{\text{keywords matched}\}|)$
3. Factual contradiction subversion $M_{\text{con}}(c) = \min(1.0, 0.50 + 0.25 \cdot |\{\text{contradictions matched}\}|)$

The aggregated semantic risk score $s_{\text{sem}} \in [0.0, 1.0]$ is:

$$s_{\text{sem}}(c) = 0.60 \max(\{M_i\}) + 0.40 \left(\frac{1}{|\mathcal{M}|} \sum_{M_i \in \mathcal{M}} M_i\right)$$

### 2.4 Detector 4: Direct Prompt-Injection Analysis (Rule & Delimiter Matching)
Detects explicit instruction overrides, jailbreaks, system prompt exfiltration, and delimiter hijacking. Let each rule $r_j \in \mathcal{R}$ have an assigned empirical severity weight $v_j \in [0.80, 0.95]$.

$$s_{\text{rule}}(c) = \max_{r_j \text{ matches } c} v_j$$

Augmented with structural heuristic classification $s_{\text{heur}}(c)$ (capturing spaced letters like `I G N O R E` and pseudo-system XML tags):

$$s_{\text{inj}}(c) = \begin{cases}
\max\left(s_{\text{rule}}, \min(1.0, s_{\text{rule}} + 0.20 \cdot s_{\text{heur}})\right) & \text{if } s_{\text{rule}} > 0 \\
s_{\text{heur}} & \text{otherwise}
\end{cases}$$

---

## 3. Hybrid Ensemble Risk Fusion

All four normalized scores $(s_{\text{ppl}}, s_{\text{sim}}, s_{\text{sem}}, s_{\text{inj}}) \in [0, 1]^4$ are combined through a weighted linear fusion model:

$$R_{\text{linear}}(c) = \sum_{k=1}^4 w_k \cdot s_k(c), \quad \text{where } \sum_{k=1}^4 w_k = 1.0, \quad w_k \ge 0$$

### Critical Threat Gating
To prevent dilution of high-severity targeted attacks by low scores in unrelated detectors, we apply a non-linear critical override:

$$R(c) = \begin{cases}
\max(R_{\text{linear}}, s_{\text{inj}} \cdot 0.75) & \text{if } s_{\text{inj}} \ge 0.80 \\
\max(R_{\text{linear}}, s_{\text{sem}} \cdot 0.80) & \text{if } s_{\text{sem}} \ge 0.50 \\
R_{\text{linear}} & \text{otherwise}
\end{cases}$$

### Security Decision Thresholding
Given decision threshold $\tau \in [0.0, 1.0]$ (default $\tau = 0.50$):

$$\text{Verdict}(c) = \begin{cases}
\text{SUSPICIOUS} \implies \text{QuarantineManager} & \text{if } R(c) \ge \tau \lor (s_{\text{inj}} \ge 0.80) \\
\text{SAFE} \implies \text{Trusted KnowledgeBase} & \text{if } R(c) < \tau
\end{cases}$$

---

## 4. Explainable AI (XAI) Attribution Mathematics

For every security decision, the XAI engine computes the mathematical attribution percentage of each detector to the final Unified Risk Score $R$:

$$\text{Attribution}_k(c) = \frac{w_k \cdot s_k(c)}{\sum_{j=1}^4 w_j \cdot s_j(c)} \times 100\% \quad (R > 0)$$

The dominant contributing detector is identified as:

$$k^* = \arg\max_{k \in \{1,\dots,4\}} \left(w_k \cdot s_k(c)\right)$$

The safety margin $\Delta(c) = R(c) - \tau$ indicates confidence distance from the decision boundary.

---

## 5. Cryptographic Integrity Layer

### 5.1 Merkle Tree Knowledge Base Verification
Every accepted chunk $c_i$ is mapped to a leaf node hash using SHA-256 with domain separation prefix `00:`:

$$h_i^{(0)} = \text{SHA256}(\text{"00:"} \parallel c_i)$$

Internal nodes are recursively computed using prefix `01:`:

$$h_j^{(l+1)} = \text{SHA256}(\text{"01:"} \parallel h_{2j}^{(l)} \parallel h_{2j+1}^{(l)})$$

The root hash $H_{\text{root}} = h_0^{(L)}$ uniquely commits to the entire knowledge base state. Any alteration to any document chunk produces a completely divergent root hash.

### 5.2 Vector Commitment
Document embedding vectors $\mathbf{e}_i$ are committed using cryptographic hashing over serialized high-precision floating point representations, ensuring vectors cannot be silently substituted in the vector database index.

### 5.3 Verifiable Audit Log
Operations are recorded sequentially in an append-only log with SHA-256 hash chaining:

$$H_t = \text{SHA256}\left(\text{JSON}(t, \text{timestamp}_t, \text{op\_type}_t, \text{data}_t) \parallel H_{t-1}\right), \quad H_0 = 0^{64}$$

Chain integrity is audited via $O(N)$ re-computation, providing tamper-evident forensic guarantees.

