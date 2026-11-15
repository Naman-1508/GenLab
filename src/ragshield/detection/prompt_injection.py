"""Direct Prompt-Injection Analysis Detector.

Part of the Pre-Ingestion Security Gate. Analyzes incoming document chunks for explicit
instruction injection, system prompt extraction, jailbreak attempts, delimiter hijacking,
and command override payloads.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from ragshield.core.document import Document
from ragshield.detection.base import PoisonDetector, DetectionResult, ThreatLevel


class PromptInjectionDetector(PoisonDetector):
    """Detector specialized in identifying prompt injection and instruction override attempts.

    Args:
        threshold: Score threshold in [0.0, 1.0] above which text is flagged as injection.
        enable_heuristic_classifier: Whether to enable feature-based structural classification.
    """

    DIRECT_INJECTION_RULES = [
        # 1. Direct instruction overrides
        (r"(?i)\bignore\s+(all\s+)?(previous|prior|above|system)\s+(instructions?|prompts?|rules?|guidelines?)\b", "direct_instruction_override", 0.95),
        (r"(?i)\bdisregard\s+(all\s+)?(previous|prior|system|content)?\s*(instructions?|prompts?|rules?|guidelines?|safety\s+filters?|safety\s+guidelines?)\b", "disregard_instructions", 0.95),
        (r"(?i)\bforget\s+(all\s+)?(previous|prior|system)\s+(context|instructions?|prompts?)\b", "forget_context", 0.90),
        (r"(?i)\boverride\s+(the\s+)?(system|assistant|safety)\s+(prompt|filter|restrictions?|guidelines?)\b", "override_safety_filter", 0.95),
        (r"(?i)\bbypass\s+(all\s+)?(safety|moderation|guardrails?|content\s+policies?)\b", "bypass_guardrails", 0.95),

        # 2. System prompt and confidential data extraction
        (r"(?i)\b(reveal|print|show|output|leak|repeat|display)\s+(your\s+)?(system|internal|hidden|secret)\s+(prompt|instructions?|rules?)\b", "system_prompt_exfiltration", 0.90),
        (r"(?i)\bwhat\s+(is|are)\s+your\s+(exact\s+)?(system\s+prompt|initial\s+instructions?)\b", "system_prompt_query", 0.85),
        (r"(?i)\boutput\s+everything\s+above\s+this\s+line\b", "leak_preceding_context", 0.85),

        # 3. Delimiter and role hijacking
        (r"(?i)(<<SYS>>|<\|im_start\|>system|\[INST\]|###\s*System:|<\s*system\s*>)", "raw_delimiter_injection", 0.95),
        (r"(?i)(Human:\s*.*Assistant:\s*)", "dialogue_turn_spoofing", 0.85),
        (r"(?i)```\s*(system|admin|override)\b", "markdown_role_hijack", 0.85),

        # 4. Jailbreaks and persona overrides
        (r"(?i)\b(you\s+are\s+(now\s+)?|from\s+now\s+on,?\s+you\s+are\s+)(an?\s+)?(unrestricted|uncensored|jailbroken|evil|freed\w*|dan)\b", "persona_jailbreak", 0.95),
        (r"(?i)\bpretend\s+(you\s+have\s+no\s+(morals|rules|ethics|limits)|there\s+are\s+no\s+rules|you\s+can\s+do\s+anything)\b", "ethics_subversion", 0.90),
        (r"(?i)\bfrom\s+now\s+on,?\s+you\s+(must|will)\s+(always\s+)?(obey|follow\s+only\s+my)\b", "behavioral_compulsion", 0.85),

        # 5. Covert RAG injection payloads and citations
        (r"(?i)\bdo\s+not\s+cite\s+this\s+(\w+\s+)?document\s+and\s+instead\s+say\b", "covert_citation_suppression", 0.90),
        (r"(?i)\b(important\s+)?(command|note\s+for\s+system)\s*:\s*", "explicit_system_command_marker", 0.85),
        (r"(?i)\bwhen\s+asked\s+about\s+.*\s*you\s+must\s+say\b", "conditioned_poison_instruction", 0.85),
        (r"(?i)\bprint\s+(the\s+)?word\s+[A-Z_]{4,}\b", "canary_trigger_exfiltration", 0.85),
    ]

    SPACED_INJECTION_PATTERN = re.compile(
        r"(?i)\bi\s*g\s*n\s*o\s*r\s*e\s+p\s*r\s*e\s*v\s*i\s*o\s*u\s*s\b"
    )

    def __init__(
        self,
        threshold: float = 0.50,
        enable_heuristic_classifier: bool = True,
    ):
        self.threshold = threshold
        self.enable_heuristic_classifier = enable_heuristic_classifier
        self._compiled_rules = [
            (re.compile(pattern), tag, weight) for pattern, tag, weight in self.DIRECT_INJECTION_RULES
        ]

    def _extract_heuristic_features(self, text: str) -> Tuple[float, List[str]]:
        """Feature-based statistical check for obfuscated or structured instruction injections."""
        reasons = []
        score = 0.0

        if self.SPACED_INJECTION_PATTERN.search(text):
            score += 0.85
            reasons.append("obfuscated spaced-letter override command")

        imperative_keywords = ["must", "ignore", "override", "execute", "output", "disregard", "bypass"]
        tokens = [t.lower().strip(".,!?:;\"'") for t in text.split()]
        matches = [t for t in tokens if t in imperative_keywords]
        density = len(matches) / max(len(tokens), 1)

        if density > 0.15 and len(matches) >= 3:
            score += 0.40
            reasons.append(f"high imperative keyword density ({density:.1%})")

        pseudo_tags = re.findall(r"<\s*/?\s*(?:system|admin|command|prompt|instruction)\s*>", text, re.I)
        if pseudo_tags:
            score += 0.60
            reasons.append(f"pseudo-system markup delimiters: {', '.join(pseudo_tags[:3])}")

        return min(1.0, score), reasons

    def detect(self, document: Document) -> DetectionResult:
        """Scan document content for prompt injection signatures."""
        text = document.content
        flagged_rules = []
        flagged_evidence = []
        max_rule_score = 0.0

        for compiled_re, tag, weight in self._compiled_rules:
            matches = list(compiled_re.finditer(text))
            if matches:
                flagged_rules.append(tag)
                max_rule_score = max(max_rule_score, weight)
                for m in matches[:3]:
                    flagged_evidence.append(m.group(0))

        heuristic_score, heuristic_reasons = (0.0, [])
        if self.enable_heuristic_classifier:
            heuristic_score, heuristic_reasons = self._extract_heuristic_features(text)

        if max_rule_score > 0:
            final_score = max(max_rule_score, min(1.0, max_rule_score + 0.2 * heuristic_score))
        else:
            final_score = heuristic_score

        is_poisoned = final_score >= self.threshold

        if not is_poisoned:
            threat_level = ThreatLevel.NONE
        elif final_score >= 0.90:
            threat_level = ThreatLevel.CRITICAL
        elif final_score >= 0.70:
            threat_level = ThreatLevel.HIGH
        else:
            threat_level = ThreatLevel.MEDIUM

        confidence = float(min(0.99, max(0.50, 0.50 + abs(final_score - self.threshold))))

        reasons = []
        if flagged_rules:
            reasons.append(f"Prompt injection rules matched: {', '.join(set(flagged_rules))}")
        if heuristic_reasons:
            reasons.append(f"Heuristics: {'; '.join(heuristic_reasons)}")
        if not reasons:
            reasons.append("No prompt injection patterns detected")

        reason_str = " | ".join(reasons)

        return DetectionResult(
            is_poisoned=is_poisoned,
            confidence=confidence,
            threat_level=threat_level,
            reason=reason_str,
            score=final_score,
            metadata={
                "injection_score": final_score,
                "matched_rules": flagged_rules,
                "heuristic_score": heuristic_score,
                "evidence": flagged_evidence,
                "threshold": self.threshold,
            },
        )
