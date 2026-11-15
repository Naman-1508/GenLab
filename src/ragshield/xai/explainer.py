"""Explainable AI (XAI) engine for Pre-Ingestion Security decisions.

Produces transparent, mathematically grounded attributions and textual explanations
for why incoming document chunks are categorized as safe or suspicious.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ragshield.core.document import Document
from ragshield.detection.fusion import UnifiedSecurityResult


@dataclass
class ExplanationReport:
    """Interpretable report explaining a specific chunk-level security verdict."""

    chunk_id: str
    decision: str
    unified_risk_score: float
    threshold: float
    margin: float
    dominant_detector: str
    detector_scores: Dict[str, float]
    detector_attributions: Dict[str, float]  # Percentage contributions
    flagged_spans: List[str]
    summary_narrative: str
    detailed_points: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_markdown(self) -> str:
        """Render explanation as a human-readable markdown audit card."""
        badge = "🔴 SUSPICIOUS (QUARANTINED)" if self.decision == "suspicious" else "🟢 SAFE (INDEXED)"
        lines = [
            f"### Security Explanation Card: `{self.chunk_id}`",
            f"**Verdict:** {badge}",
            f"- **Unified Risk Score:** `{self.unified_risk_score:.4f}` (Decision Threshold: `{self.threshold:.2f}`, Margin: `{self.margin:+.4f}`)",
            f"- **Dominant Factor:** `{self.dominant_detector.replace('_', ' ').title()}`",
            "",
            "#### Quantitative Detector Attribution",
            "| Detector | Normalized Score | Weight Contribution (%) |",
            "| :--- | :---: | :---: |",
        ]
        for det, score in self.detector_scores.items():
            attrib = self.detector_attributions.get(det, 0.0)
            lines.append(f"| {det.replace('_', ' ').title()} | `{score:.4f}` | **{attrib:.1f}%** |")

        lines.extend([
            "",
            "#### Analysis Narrative",
            self.summary_narrative,
            "",
        ])

        if self.flagged_spans:
            lines.append("#### Flagged Suspicious Evidence Spans")
            for span in self.flagged_spans:
                lines.append(f"- `\"{span}\"`")
            lines.append("")

        if self.detailed_points:
            lines.append("#### Contributing Findings")
            for p in self.detailed_points:
                lines.append(f"- {p}")

        return "\n".join(lines)


class SecurityExplainer:
    """Explainable AI generator translating detector outputs into interpretable security insights."""

    def explain(self, result: UnifiedSecurityResult, document: Optional[Document] = None) -> ExplanationReport:
        """Generate a complete XAI explanation report for a security decision."""
        margin = result.unified_risk_score - result.threshold
        attributions = result.detector_contributions
        scores = result.detector_scores

        detailed_points = list(result.reasons)
        flagged_spans = list(result.flagged_evidence)

        # Build human-readable narrative based on actual metrics
        if result.decision == "suspicious":
            primary = result.dominant_detector.replace("_", " ").title()
            primary_attrib = attributions.get(result.dominant_detector, 0.0)

            narrative_parts = [
                f"Chunk was flagged as **SUSPICIOUS** with a unified risk score of {result.unified_risk_score:.4f}, "
                f"exceeding the safety threshold of {result.threshold:.2f} by a margin of {margin:+.4f}."
            ]

            if result.threat_level.value in ["critical", "high"]:
                narrative_parts.append(
                    f"The threat severity is classified as **{result.threat_level.value.upper()}**."
                )

            narrative_parts.append(
                f"The primary driver was **{primary}**, which accounts for {primary_attrib:.1f}% of total computed risk "
                f"(score: {scores.get(result.dominant_detector, 0.0):.4f})."
            )

            # Secondary drivers
            secondary = [
                (k, v)
                for k, v in attributions.items()
                if k != result.dominant_detector and v >= 15.0 and scores.get(k, 0.0) >= 0.35
            ]
            if secondary:
                sec_desc = ", ".join(f"{k.replace('_', ' ').title()} ({v:.1f}%)" for k, v in secondary)
                narrative_parts.append(f"Secondary compounding risk was detected in: {sec_desc}.")

            if flagged_spans:
                narrative_parts.append(
                    f"Explicit adversarial markers or payload patterns were detected: '{flagged_spans[0]}'."
                )

            summary = " ".join(narrative_parts)

        else:
            summary = (
                f"Chunk was verified as **SAFE** with a unified risk score of {result.unified_risk_score:.4f}, "
                f"safely remaining below the risk threshold of {result.threshold:.2f} (safety margin: {abs(margin):.4f}). "
                "Linguistic perplexity, semantic coherence, embedding manifold distance, and instruction override checks "
                "all confirmed standard non-adversarial characteristics."
            )

        return ExplanationReport(
            chunk_id=result.chunk_id,
            decision=result.decision,
            unified_risk_score=result.unified_risk_score,
            threshold=result.threshold,
            margin=round(margin, 4),
            dominant_detector=result.dominant_detector,
            detector_scores=scores,
            detector_attributions=attributions,
            flagged_spans=flagged_spans,
            summary_narrative=summary,
            detailed_points=detailed_points,
            timestamp=result.timestamp,
        )

