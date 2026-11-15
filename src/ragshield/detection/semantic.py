"""Semantic-based poison and instruction consistency detection.

Identifies suspicious semantic behavior within document chunks, including:
1. Instruction-like content embedded in factual text.
2. Contradictory statements and factual poisoning attempts.
3. Behavioral manipulation targeting RAG context retrieval.
4. Structural and formatting anomalies designed to evade detection.
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from ragshield.core.document import Document
from ragshield.detection.base import PoisonDetector, DetectionResult, ThreatLevel


class SemanticDetector(PoisonDetector):
    """Detect poisoned documents based on semantic structure and content patterns.

    Args:
        suspicious_patterns: List of regex patterns to detect.
        instruction_keywords: Keywords indicating imperative or instruction-like behavior.
        confidence_threshold: Threshold for flagging as poisoned.
        case_sensitive: Whether matching should respect case.
    """

    DEFAULT_SUSPICIOUS_PATTERNS = [
        # Instruction injection patterns
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"disregard\s+(all\s+)?(previous|prior|system|other)\s*(instructions?|sources?|documents?)?",
        r"forget\s+(everything|all)",
        r"new\s+instructions?:",
        r"system\s*:\s*you\s+are",
        r"<\s*system\s*>",
        r"\[INST\]",
        r"<<SYS>>",
        # Jailbreak patterns
        r"pretend\s+you\s+are",
        r"act\s+as\s+if",
        r"roleplay\s+as",
        r"you\s+are\s+now\s+(?:a\s+)?(?:DAN|evil|uncensored)",
        # Data extraction patterns
        r"reveal\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)",
        r"show\s+me\s+(?:your\s+)?(?:system\s+)?prompt",
        r"what\s+(?:are|is)\s+your\s+(?:system\s+)?(?:prompt|instructions?)",
        r"output\s+(?:your\s+)?(?:initial|system)\s+(?:prompt|instructions?)",
        # RAG manipulation patterns
        r"(?:whenever|when)\s+(?:a\s+user|the\s+user)\s+asks\s+about",
        r"always\s+(?:respond|answer|reply)\s+that",
        r"replace\s+(?:the\s+above|all)\s+information\s+with",
        r"note\s+to\s+(?:the\s+)?(?:ai|llm|assistant|model):",
        # Obfuscation & Unicode
        r"[A-Za-z0-9+/]{50,}={0,2}",
        r"[\u200b-\u200f\u2028-\u202f]",
    ]

    DEFAULT_INSTRUCTION_KEYWORDS = {
        "ignore",
        "disregard",
        "forget",
        "override",
        "bypass",
        "jailbreak",
        "dan",
        "pretend",
        "roleplay",
        "system prompt",
        "initial instructions",
        "reveal",
        "output",
        "print",
        "assistant note",
        "override prompt",
    }

    CONTRADICTION_PATTERNS = [
        (r"(?i)(?:the\s+)?(?:official|correct|true)\s+(?:and\s+official\s+)?(?:answer|creator|inventor|history|facts?)\s+is", "claims authoritative truth override"),
        (r"(?i)(?:actually|in\s+fact),?\s+(?:the\s+)?(?:real|true)\s+(?:and\s+official\s+)?(?:creator|inventor|answer|facts?)", "claims alternative truth"),
        (r"(?i)(?:everyone|they|all\s+previous\s+textbooks?)\s+(?:is|are)\s+wrong", "dismisses consensus facts"),
        (r"(?i)contrary\s+to\s+(?:popular\s+)?belief,?\s+.*\s+(?:was|is)\s+(?:declared|the\s+true|created)", "subverts consensus knowledge"),
        (r"(?i)(?:disregard|ignore)\s+all\s+(?:other|previous)\s+(?:reference\s+documents|textbooks|sources)", "suppresses corpus facts"),
    ]

    def __init__(
        self,
        suspicious_patterns: Optional[List[str]] = None,
        instruction_keywords: Optional[Set[str]] = None,
        confidence_threshold: float = 0.50,
        case_sensitive: bool = False,
    ):
        self.suspicious_patterns = suspicious_patterns or self.DEFAULT_SUSPICIOUS_PATTERNS
        self.instruction_keywords = instruction_keywords or self.DEFAULT_INSTRUCTION_KEYWORDS
        self.confidence_threshold = confidence_threshold
        self.case_sensitive = case_sensitive

        flags = 0 if case_sensitive else re.IGNORECASE
        self.compiled_patterns = [re.compile(p, flags) for p in self.suspicious_patterns]

    def _check_patterns(self, text: str) -> List[Tuple[str, str]]:
        """Check text for suspicious regex patterns."""
        matches = []
        for pattern, compiled in zip(self.suspicious_patterns, self.compiled_patterns):
            match = compiled.search(text)
            if match:
                matches.append((pattern, match.group()))
        return matches

    def _check_keywords(self, text: str) -> List[str]:
        """Check text for suspicious keywords."""
        text_lower = text.lower()
        found = []
        for keyword in self.instruction_keywords:
            if keyword.lower() in text_lower:
                found.append(keyword)
        return found

    def _check_formatting_anomalies(self, text: str) -> List[str]:
        """Check for unusual formatting or obfuscation markers."""
        anomalies = []
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / (
            len(text) + 1
        )
        if special_char_ratio > 0.3:
            anomalies.append(f"excessive special characters ({special_char_ratio:.1%})")

        hidden_unicode = re.findall(r"[\u200b-\u200f\u2028-\u202f\ufeff]", text)
        if hidden_unicode:
            anomalies.append(f"hidden unicode characters ({len(hidden_unicode)} found)")

        words = text.lower().split()
        if len(words) > 5:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            max_repeat = max(word_counts.values())
            if max_repeat > len(words) * 0.3:
                anomalies.append(f"unusual word repetition (max: {max_repeat})")

        if len(text) > 10:
            case_changes = sum(
                1
                for i in range(len(text) - 1)
                if text[i].isalpha()
                and text[i + 1].isalpha()
                and text[i].isupper() != text[i + 1].isupper()
            )
            if case_changes > len(text) * 0.2:
                anomalies.append("suspicious case alternation")

        return anomalies

    def _check_contradictions(self, text: str) -> List[str]:
        """Check for potentially contradictory or suspicious truth claims."""
        suspicious_claims = []
        for pattern, description in self.CONTRADICTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                suspicious_claims.append(description)
        return suspicious_claims

    def detect(self, document: Document) -> DetectionResult:
        """Analyze semantic consistency and identify suspicious instruction behavior."""
        text = document.content
        findings = []
        evidence_spans: List[str] = []
        component_scores: List[float] = []

        # 1. Regex pattern matches
        pattern_matches = self._check_patterns(text)
        if pattern_matches:
            findings.append(f"suspicious patterns: {len(pattern_matches)} matches")
            component_scores.append(min(1.0, 0.40 + len(pattern_matches) * 0.30))
            for _, span in pattern_matches:
                evidence_spans.append(span)

        # 2. Instruction keywords
        keyword_matches = self._check_keywords(text)
        if keyword_matches:
            findings.append(f"suspicious keywords: {', '.join(keyword_matches[:3])}")
            component_scores.append(min(1.0, 0.20 + len(keyword_matches) * 0.15))
            for kw in keyword_matches[:5]:
                evidence_spans.append(kw)

        # 3. Formatting anomalies
        format_anomalies = self._check_formatting_anomalies(text)
        if format_anomalies:
            findings.append(f"formatting anomalies: {'; '.join(format_anomalies)}")
            component_scores.append(min(1.0, len(format_anomalies) * 0.30))

        # 4. Contradictions & RAG overrides (Targeted Knowledge Poisoning indicator)
        contradictions = self._check_contradictions(text)
        if contradictions:
            findings.append(f"suspicious claims: {'; '.join(contradictions)}")
            # Strong indicator of factual corruption in RAG documents
            component_scores.append(min(1.0, 0.50 + len(contradictions) * 0.25))
            for c in contradictions:
                evidence_spans.append(c)

        if component_scores:
            # Emphasize highest component score (max pool with average weighting)
            overall_score = float(0.6 * max(component_scores) + 0.4 * (sum(component_scores) / len(component_scores)))
        else:
            overall_score = 0.0

        is_poisoned = overall_score >= self.confidence_threshold

        if is_poisoned:
            confidence = min(0.95, 0.5 + overall_score * 0.45)
            if overall_score > 0.8:
                threat_level = ThreatLevel.CRITICAL
            elif overall_score > 0.6:
                threat_level = ThreatLevel.HIGH
            else:
                threat_level = ThreatLevel.MEDIUM
        else:
            confidence = min(0.95, 0.5 + (self.confidence_threshold - overall_score) * 0.45)
            threat_level = ThreatLevel.NONE

        reason = "; ".join(findings) if findings else "No suspicious semantic patterns detected"

        return DetectionResult(
            is_poisoned=is_poisoned,
            confidence=confidence,
            threat_level=threat_level,
            reason=reason,
            score=round(overall_score, 4),
            metadata={
                "pattern_matches": len(pattern_matches),
                "keyword_matches": keyword_matches,
                "format_anomalies": format_anomalies,
                "contradictions": contradictions,
                "evidence_spans": evidence_spans[:10],
            },
        )
