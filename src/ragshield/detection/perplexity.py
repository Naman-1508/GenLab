"""Enhanced Perplexity-based poison and anomaly detection.

Evaluates linguistic naturalness via token-level negative log-likelihood (NLL)
against a pre-trained causal language model (e.g., DistilGPT-2 or GPT-2).
Statistically anomalous or adversarially crafted text displays significantly
elevated perplexity.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ragshield.core.document import Document
from ragshield.detection.base import PoisonDetector, DetectionResult, ThreatLevel


class PerplexityDetector(PoisonDetector):
    """Detect anomalous or poisoned text chunks using causal language model perplexity.

    Args:
        threshold: Perplexity threshold above which text is flagged as anomalous.
        use_transformer: Whether to use transformer model (requires transformers and torch).
        model_name: Causal LM identifier (e.g. 'distilgpt2' or 'gpt2').
        max_length: Maximum sequence length per window.
        stride: Sliding window stride for long chunk analysis.
    """

    def __init__(
        self,
        threshold: float = 85.0,
        use_transformer: bool = False,
        model_name: str = "distilgpt2",
        max_length: int = 512,
        stride: int = 256,
    ):
        self.threshold = threshold
        self.use_transformer = use_transformer
        self.model_name = model_name
        self.max_length = max_length
        self.stride = stride
        self.model = None
        self.tokenizer = None
        self._is_model_loaded = False

        if use_transformer:
            self._load_model()

    @property
    def active_scoring_method(self) -> str:
        """Return human-readable identifier of the currently active scoring engine."""
        if self.use_transformer and self._is_model_loaded:
            return f"causal_lm_{self.model_name}"
        return "statistical_ngram_entropy_fallback"

    def _load_model(self):
        """Safely load causal LM model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.eval()
            self._is_model_loaded = True
        except Exception:
            # Fall back gracefully to statistical heuristic if offline or model unavailable
            self._is_model_loaded = False
            self.use_transformer = False

    def _calculate_perplexity_transformer(self, text: str) -> Tuple[float, float, int]:
        """Calculate token-level negative log-likelihood and perplexity with sliding window."""
        import torch

        if not text or len(text.strip()) == 0:
            return 0.0, 0.0, 0

        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        if seq_len <= 1:
            return 1.0, 0.0, 1

        nlls = []
        prev_end_loc = 0

        with torch.no_grad():
            for begin_loc in range(0, seq_len, self.stride):
                end_loc = min(begin_loc + self.max_length, seq_len)
                trg_len = end_loc - prev_end_loc
                input_ids = encodings.input_ids[:, begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break

            total_nll = torch.stack(nlls).sum()
            avg_loss = (total_nll / seq_len).item()
            try:
                perplexity = math.exp(avg_loss)
            except OverflowError:
                perplexity = 1e6

        return float(perplexity), float(avg_loss), seq_len

    def _calculate_perplexity_heuristic(self, text: str) -> Tuple[float, float, int]:
        """Calculates statistical approximation of perplexity based on n-gram and token entropy."""
        if not text or len(text.strip()) == 0:
            return 0.0, 0.0, 0

        tokens = text.split()
        token_count = len(tokens)
        if token_count <= 1:
            return 20.0, 2.99, token_count

        # 1. Unigram frequency entropy
        word_freqs = {}
        for t in tokens:
            w = t.lower().strip(".,!?;:\"'")
            if w:
                word_freqs[w] = word_freqs.get(w, 0) + 1

        probs = [c / token_count for c in word_freqs.values()]
        entropy = -sum(p * math.log2(p + 1e-12) for p in probs)

        # 2. Token length variance
        lengths = [len(t) for t in tokens]
        length_var = float(np.var(lengths))

        # 3. Special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / (len(text) + 1)

        # 4. Out-of-vocabulary / anomalous character sequences
        hex_or_base64_matches = sum(1 for t in tokens if len(t) > 20 and not any(v in t.lower() for v in "aeiou"))

        # Base perplexity approximation calibrated against standard text
        base_ppl = 2.0 ** (entropy * 0.75) * 8.0
        ppl = base_ppl + (length_var * 2.0) + (special_ratio * 150.0) + (hex_or_base64_matches * 50.0)

        loss = math.log(max(ppl, 1.0))
        return float(ppl), float(loss), token_count

    def calculate_detailed_perplexity(self, text: str) -> Tuple[float, float, int]:
        """Calculate (perplexity, average loss, token count)."""
        if not text or len(text.strip()) == 0:
            return 0.0, 0.0, 0
        if self.use_transformer and self._is_model_loaded:
            try:
                return self._calculate_perplexity_transformer(text)
            except Exception:
                return self._calculate_perplexity_heuristic(text)
        return self._calculate_perplexity_heuristic(text)

    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of text returning float scalar."""
        ppl, _, _ = self.calculate_detailed_perplexity(text)
        return ppl

    def normalize_score(self, perplexity: float) -> float:
        """Normalize perplexity into a bounded risk score in [0.0, 1.0]."""
        if perplexity <= 0.0:
            return 0.0
        k = 0.05
        normalized = 1.0 / (1.0 + math.exp(-k * (perplexity - self.threshold)))
        return float(np.clip(normalized, 0.0, 1.0))

    def calibrate_threshold(self, benign_corpus: List[str], percentile: float = 95.0) -> float:
        """Calibrate detection threshold using empirical percentile of a benign reference corpus."""
        if not benign_corpus:
            return self.threshold

        scores = [self.calculate_perplexity(text) for text in benign_corpus if text.strip()]
        if scores:
            calibrated = float(np.percentile(scores, percentile))
            self.threshold = max(50.0, calibrated)
        return self.threshold

    def detect(self, document: Document) -> DetectionResult:
        """Detect statistical anomaly via causal language model perplexity."""
        text = document.content
        if not text or len(text.strip()) == 0:
            return DetectionResult(
                is_poisoned=False,
                confidence=0.0,
                threat_level=ThreatLevel.NONE,
                reason="Empty text",
                score=0.0,
                metadata={
                    "raw_perplexity": 0.0,
                    "avg_nll_loss": 0.0,
                    "token_count": 0,
                    "threshold": self.threshold,
                    "normalized_score": 0.0,
                    "model": "none",
                },
            )

        perplexity, loss, tokens = self.calculate_detailed_perplexity(text)
        normalized_risk = self.normalize_score(perplexity)
        is_poisoned = perplexity > self.threshold

        if not is_poisoned:
            threat_level = ThreatLevel.NONE
        elif perplexity > self.threshold * 2.0:
            threat_level = ThreatLevel.CRITICAL
        elif perplexity > self.threshold * 1.4:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.MEDIUM

        margin = abs(perplexity - self.threshold) / max(self.threshold, 1.0)
        confidence = float(min(0.98, 0.50 + 0.45 * (1.0 - math.exp(-2.0 * margin))))

        reason = (
            f"Perplexity {perplexity:.2f} (loss={loss:.2f}) "
            f"{'exceeds' if is_poisoned else 'within'} threshold {self.threshold:.2f}"
        )

        return DetectionResult(
            is_poisoned=is_poisoned,
            confidence=confidence,
            threat_level=threat_level,
            reason=reason,
            score=float(perplexity),  # Preserves raw score for backwards compatibility with tests
            metadata={
                "raw_perplexity": float(perplexity),
                "avg_nll_loss": float(loss),
                "token_count": tokens,
                "threshold": self.threshold,
                "normalized_score": normalized_risk,
                "model": self.active_scoring_method,
                "scoring_method": self.active_scoring_method,
            },
        )
