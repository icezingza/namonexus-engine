"""
Explainability Layer (XAI) — Feature 4.1
==========================================
Patent-Pending Technology | NamoNexus Research Team

Problem
-------
Regulated industries — healthcare, HR screening, legal risk assessment,
and financial fraud detection — require an *audit trail* explaining why
the fusion engine produced a specific risk classification.  Without this,
operators cannot answer basic questions such as:

  "Which sensor drove the high-risk flag at 14:32?"
  "Why did the system overrule the voice score?"
  "Is the face modality systematically over-weighted for this subject?"

Absent explainability, the engine cannot pass PDPA, GDPR, or FDA
compliance requirements.

Solution: Shapley-Value Attribution with Golden Ratio Prior Weighting
----------------------------------------------------------------------
We adapt the classic Shapley value framework from cooperative game theory
to explain each Bayesian update.  In this context:

  Players  = modalities (text, voice, face, ...)
  Payoff   = change in fused_score caused by a coalition of updates

For computational tractability in the streaming setting, we use the
*marginal contribution* approximation, which is exact for independent
modalities and a tight bound otherwise:

    phi_m = w_m * (score_m - fused_score_before)

where w_m is the effective weight of modality m, composed of:
    1. Raw confidence (provided by the caller)
    2. Trust calibration weight (from Phase 2, if available)
    3. Golden Ratio prior contribution fraction

The **Golden Ratio prior contribution** is the key patent element: the
prior alpha0/beta0 = phi contributes a baseline "null attribution" that
anchors every explanation to the Golden Ratio.  Modalities are attributed
relative to this prior:

    prior_fraction_m = phi_prior / (phi_prior + sum_of_modality_weights)

where phi_prior = alpha0 (the prior strength anchored at phi).

Output Format
-------------
Every call to engine.explain() returns an ExplanationReport containing:
  - Per-modality Shapley attributions (signed, normalized to [-1, +1])
  - Confidence-weighted contribution percentages
  - Prior attribution (the "default" baseline)
  - Human-readable risk narrative (per-modality, one sentence each)
  - GDPR/PDPA-compliant summary text suitable for logging

Patent Claim (new — Claim 13)
------------------------------
"A method of generating human-readable explanations for the output of a
multimodal Bayesian fusion engine, comprising:
(a) computing per-modality marginal attributions as Shapley values,
    wherein each attribution is the product of the modality's effective
    confidence weight and its marginal score contribution relative to
    the pre-update fused posterior;
(b) normalizing attributions relative to a Golden Ratio prior baseline
    attribution, defined as the contribution of the initial prior
    alpha0/beta0 = phi to the total accumulated evidence;
(c) generating a per-modality human-readable explanation sentence that
    states the modality name, its normalized attribution, its raw score,
    and its alignment with the aggregate posterior (consistent / mixed /
    contradicting);
(d) assembling a structured ExplanationReport that satisfies GDPR
    Article 22 and PDPA Section 40 audit trail requirements;
such that each risk classification produced by the fusion engine is
traceable to specific sensor contributions."
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Golden Ratio constants (local copies — zero external dependency)
_PHI: float        = (1.0 + 5.0 ** 0.5) / 2.0   # ≈ 1.618
_PHI_RECIP: float  = 1.0 / _PHI                   # ≈ 0.618


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ModalityAlignment(Enum):
    """How well a modality's score aligned with the aggregate posterior."""
    CONSISTENT    = "consistent"     # |score_m - fused| < low_threshold
    MIXED         = "mixed"          # low_threshold <= |score_m - fused| < high_threshold
    CONTRADICTING = "contradicting"  # |score_m - fused| >= high_threshold


class RiskNarrative(Enum):
    """Human-readable risk narrative levels."""
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModalityAttribution:
    """
    Shapley attribution for one modality in a single fusion step.

    Fields
    ------
    modality:
        Sensor name (e.g., "text", "voice", "face").
    raw_score:
        The score provided by this modality (in [0, 1]).
    raw_confidence:
        The confidence provided by this modality (in [0, 1]).
    effective_weight:
        Composite weight = confidence × trust_calibration_factor.
    marginal_contribution:
        Signed marginal contribution to the fused score delta:
        phi_m = effective_weight × (raw_score − fused_before).
    normalized_attribution:
        Attribution normalized to [-1, +1] relative to total attribution
        magnitude (including prior).
    contribution_pct:
        Percentage of total positive attribution this modality contributed.
        Negative attributions count as 0% (they reduce the score).
    alignment:
        How well this modality's score aligned with the aggregate.
    explanation_sentence:
        One-sentence human-readable explanation for compliance logs.
    """

    modality:               str
    raw_score:              float
    raw_confidence:         float
    effective_weight:       float
    marginal_contribution:  float
    normalized_attribution: float
    contribution_pct:       float
    alignment:              ModalityAlignment
    explanation_sentence:   str

    @property
    def weighted_shapley(self) -> float:
        """Alias for normalized_attribution (benchmark compatibility)."""
        return self.normalized_attribution

    @property
    def shapley_value(self) -> float:
        """Legacy alias used by older examples/documentation."""
        return self.normalized_attribution

    @property
    def phi_weight(self) -> float:
        """Alias for effective_weight (benchmark compatibility)."""
        return self.effective_weight

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["alignment"] = self.alignment.value
        d["weighted_shapley"] = self.weighted_shapley
        d["shapley_value"] = self.shapley_value
        d["phi_weight"] = self.phi_weight
        return d


@dataclass
class PriorAttribution:
    """
    Attribution of the Golden Ratio prior to the current fused score.

    The prior always contributes a baseline — even before any observation.
    This attribution quantifies how much of the current posterior is
    "inherited" from the prior versus driven by observations.
    """

    alpha0:             float   # Prior alpha (anchored at phi * prior_strength)
    beta0:              float   # Prior beta
    prior_mean:         float   # alpha0 / (alpha0 + beta0)
    phi_ratio:          float   # alpha0 / beta0 (should be near phi)
    prior_fraction:     float   # Fraction of total evidence from prior
    prior_strength:     float   # alpha0 + beta0 (total prior mass)
    explanation_sentence: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplanationReport:
    """
    Structured explanation for one fusion state snapshot.

    Compliant with GDPR Article 22 (automated decision-making) and
    PDPA Section 40 audit trail requirements.

    Fields
    ------
    timestamp:
        Wall-clock time of the explanation (ISO-8601 string).
    fused_score:
        The current fused posterior mean.
    fused_score_before:
        The fused score immediately before the most recent update.
    risk_level:
        Classification: low / medium / high.
    modality_attributions:
        Shapley attributions per modality.
    prior_attribution:
        Golden Ratio prior attribution.
    dominant_modality:
        The modality with the highest absolute normalized attribution.
    total_obs_count:
        Total observations processed by the engine at this point.
    summary_text:
        Human-readable paragraph for compliance logging.
    metadata:
        Optional additional context (session_id, subject_id, etc.).
    """

    timestamp:             str
    fused_score:           float
    fused_score_before:    float
    risk_level:            str
    modality_attributions: List[ModalityAttribution]
    prior_attribution:     PriorAttribution
    dominant_modality:     Optional[str]
    total_obs_count:       int
    summary_text:          str
    metadata:              Dict[str, Any] = field(default_factory=dict)

    @property
    def narrative(self) -> str:
        """Alias for summary_text (benchmark compatibility)."""
        return self.summary_text

    def top_modality(self) -> Optional[ModalityAttribution]:
        """Return the ModalityAttribution for the dominant modality."""
        if not self.dominant_modality:
            return None
        for m in self.modality_attributions:
            if m.modality == self.dominant_modality:
                return m
        return None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["modality_attributions"] = [m.to_dict() for m in self.modality_attributions]
        d["prior_attribution"]     = self.prior_attribution.to_dict()
        d["narrative"] = self.narrative
        return d

    def to_audit_dict(self) -> Dict[str, Any]:
        """Alias for to_dict() with additional audit metadata."""
        d = self.to_dict()
        d["audit_version"] = "1.0.0"
        return d

    def __repr__(self) -> str:
        lines = [
            f"ExplanationReport | {self.timestamp}",
            f"  fused_score : {self.fused_score:.4f}  ({self.risk_level} risk)",
            f"  delta       : {self.fused_score - self.fused_score_before:+.4f}",
            f"  dominant    : {self.dominant_modality}",
        ]
        for m in self.modality_attributions:
            lines.append(
                f"  [{m.modality:<8}] score={m.raw_score:.3f}  "
                f"attr={m.normalized_attribution:+.4f}  "
                f"({m.alignment.value})"
            )
        lines.append(f"  [prior   ] fraction={self.prior_attribution.prior_fraction:.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# XAI Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplanationConfig:
    """
    Configuration for the ExplainabilityLayer.

    Parameters
    ----------
    consistent_threshold:
        |score_m - fused| < this → CONSISTENT alignment.
        Default = 1/phi^2 ≈ 0.382.
    contradicting_threshold:
        |score_m - fused| >= this → CONTRADICTING alignment.
        Default = 1/phi ≈ 0.618.
    min_contribution_pct:
        Modalities contributing less than this % are described as
        "negligible" in the narrative.
    include_prior_in_narrative:
        If True, include the prior attribution sentence in summary_text.
    risk_thresholds:
        (low_upper, high_lower) boundaries for risk classification.
        Scores above low_upper = low risk; below high_lower = high risk.
    """

    consistent_threshold:       float = _PHI_RECIP ** 2   # 1/phi^2 ≈ 0.382
    contradicting_threshold:    float = _PHI_RECIP         # 1/phi ≈ 0.618
    min_contribution_pct:       float = 5.0
    include_prior_in_narrative: bool  = True
    risk_thresholds:            Tuple[float, float] = (0.65, 0.40)

    @classmethod
    def default(cls) -> "ExplanationConfig":
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-update observation record (internal)
# ---------------------------------------------------------------------------


@dataclass
class _ObsRecord:
    """Internal record of one engine update, used for attribution."""
    modality:        str
    raw_score:       float
    raw_confidence:  float
    trust_factor:    float   # From Phase 2 calibrator (default 1.0)
    fused_before:    float
    fused_after:     float
    alpha_before:    float
    beta_before:     float
    alpha_after:     float
    beta_after:      float
    timestamp:       float


# ---------------------------------------------------------------------------
# Main Explainability Layer
# ---------------------------------------------------------------------------


class ShapleyExplainer:
    """
    Computes Shapley-value attributions and generates compliance-grade
    explanations for any NamoNexus fusion engine.

    Architecture
    ------------
    The ExplainabilityLayer is a **non-intrusive observer** — it is
    attached to an existing engine and records the state changes caused
    by each ``update()`` call.  It does NOT modify the engine's logic.

    Usage — composition pattern:
    ::

        engine = Phase4GoldenFusion()
        xai    = ExplainabilityLayer(engine)

        engine.update(0.85, 0.70, "text")
        engine.update(0.25, 0.90, "voice")

        report = xai.explain()
        print(report.summary_text)
        print(report)

    Or use Phase4GoldenFusion.explain() directly (built-in):
    ::

        engine = Phase4GoldenFusion()
        engine.update(0.85, 0.70, "text")
        engine.update(0.25, 0.90, "voice")
        report = engine.explain()

    Parameters
    ----------
    engine:
        Fusion engine instance.  Must expose: ``fused_score``,
        ``uncertainty``, ``risk_level``, ``_alpha``, ``_beta``,
        ``_alpha0``, ``_beta0``, ``_history``.
    config:
        XAI configuration.
    max_history:
        Maximum attribution records kept in memory.
    """

    def __init__(
        self,
        engine:      Optional[Any] = None,
        config:      Optional[XAIConfig] = None,
        max_history: int = 200,
    ) -> None:
        # Keep a safe fallback so the explainer can still operate in
        # standalone or partially wired scenarios.
        self._engine      = engine if engine is not None else object()
        self._cfg         = config or XAIConfig.default()
        self._max_history = max_history
        self._records:    List[_ObsRecord] = []
        self._reports:    List[ExplanationReport] = []

        logger.info(
            "ExplainabilityLayer | consistent_thr=%.3f contradicting_thr=%.3f",
            self._cfg.consistent_threshold,
            self._cfg.contradicting_threshold,
        )

    # ------------------------------------------------------------------
    # Record injection (called by Phase4GoldenFusion.update)
    # ------------------------------------------------------------------

    def record(
        self,
        modality:       str,
        raw_score:      float,
        raw_confidence: float,
        trust_factor:   float,
        fused_before:   float,
        fused_after:    float,
        alpha_before:   float,
        beta_before:    float,
        alpha_after:    float,
        beta_after:     float,
    ) -> None:
        """
        Record one engine update.  Called automatically by Phase4GoldenFusion.

        Parameters correspond to the engine state immediately before and
        after a single ``update()`` call.
        """
        rec = _ObsRecord(
            modality        = modality,
            raw_score       = raw_score,
            raw_confidence  = raw_confidence,
            trust_factor    = trust_factor,
            fused_before    = fused_before,
            fused_after     = fused_after,
            alpha_before    = alpha_before,
            beta_before     = beta_before,
            alpha_after     = alpha_after,
            beta_after      = beta_after,
            timestamp       = time.time(),
        )
        self._records.append(rec)
        if len(self._records) > self._max_history:
            self._records.pop(0)

    # ------------------------------------------------------------------
    # Core: compute Shapley attribution for a window of records
    # ------------------------------------------------------------------

    def _compute_attribution(
        self,
        records:    List[_ObsRecord],
        alpha0:     float,
        beta0:      float,
        fused_now:  float,
    ) -> Tuple[List[ModalityAttribution], PriorAttribution]:
        """
        Compute per-modality Shapley attributions and prior attribution
        for a set of observation records.
        """
        if not records:
            return [], self._prior_attribution(alpha0, beta0, 0.0, total_evidence=0.0)

        # Aggregate accumulated evidence per modality
        modality_data: Dict[str, Dict] = {}
        for rec in records:
            m = rec.modality
            if m not in modality_data:
                modality_data[m] = {
                    "scores":        [],
                    "confidences":   [],
                    "trust_factors": [],
                    "contributions": [],
                    "fused_before":  rec.fused_before,
                }
            d = modality_data[m]
            d["scores"].append(rec.raw_score)
            d["confidences"].append(rec.raw_confidence)
            d["trust_factors"].append(rec.trust_factor)
            # Marginal contribution = effective_weight × (score - fused_before)
            eff_weight = rec.raw_confidence * rec.trust_factor
            d["contributions"].append(eff_weight * (rec.raw_score - rec.fused_before))

        # Aggregate per modality
        mod_summaries: List[Dict] = []
        for mod, d in modality_data.items():
            mean_score  = float(np.mean(d["scores"]))
            mean_conf   = float(np.mean(d["confidences"]))
            mean_trust  = float(np.mean(d["trust_factors"]))
            eff_weight  = mean_conf * mean_trust
            marginal    = float(np.mean(d["contributions"]))
            fused_ref   = d["fused_before"]

            alignment   = self._classify_alignment(mean_score, fused_ref)
            mod_summaries.append({
                "modality":    mod,
                "mean_score":  mean_score,
                "mean_conf":   mean_conf,
                "eff_weight":  eff_weight,
                "marginal":    marginal,
                "alignment":   alignment,
                "n_obs":       len(d["scores"]),
            })

        # Prior attribution
        total_evidence = sum(
            ms["eff_weight"] for ms in mod_summaries
        )
        prior_strength = alpha0 + beta0
        prior_fraction = prior_strength / max(prior_strength + total_evidence * 10.0, 1e-9)
        prior_attr = self._prior_attribution(alpha0, beta0, prior_fraction, total_evidence)

        # Normalize attributions
        total_abs = sum(abs(ms["marginal"]) for ms in mod_summaries) + 1e-12
        total_pos = sum(max(0.0, ms["marginal"]) for ms in mod_summaries) + 1e-12

        attributions: List[ModalityAttribution] = []
        for ms in mod_summaries:
            norm_attr   = ms["marginal"] / total_abs
            contrib_pct = max(0.0, ms["marginal"]) / total_pos * 100.0
            sentence    = self._build_sentence(
                modality    = ms["modality"],
                raw_score   = ms["mean_score"],
                confidence  = ms["mean_conf"],
                norm_attr   = norm_attr,
                alignment   = ms["alignment"],
                contrib_pct = contrib_pct,
                fused_score = fused_now,
                n_obs       = ms["n_obs"],
            )
            attributions.append(ModalityAttribution(
                modality               = ms["modality"],
                raw_score              = round(ms["mean_score"], 4),
                raw_confidence         = round(ms["mean_conf"],  4),
                effective_weight       = round(ms["eff_weight"], 4),
                marginal_contribution  = round(ms["marginal"],   6),
                normalized_attribution = round(norm_attr,        4),
                contribution_pct       = round(contrib_pct,      2),
                alignment              = ms["alignment"],
                explanation_sentence   = sentence,
            ))

        # Sort by absolute attribution descending
        attributions.sort(key=lambda a: abs(a.normalized_attribution), reverse=True)
        return attributions, prior_attr

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _classify_alignment(self, score: float, fused: float) -> ModalityAlignment:
        diff = abs(score - fused)
        if diff < self._cfg.consistent_threshold:
            return ModalityAlignment.CONSISTENT
        if diff < self._cfg.contradicting_threshold:
            return ModalityAlignment.MIXED
        return ModalityAlignment.CONTRADICTING

    def _classify_risk(self, score: float) -> str:
        low_upper, high_lower = self._cfg.risk_thresholds
        if score >= low_upper:
            return "low"
        if score <= high_lower:
            return "high"
        return "medium"

    def _prior_attribution(
        self,
        alpha0:         float,
        beta0:          float,
        prior_fraction: float,
        total_evidence: float,
    ) -> PriorAttribution:
        prior_mean = alpha0 / max(alpha0 + beta0, 1e-12)
        phi_ratio  = alpha0 / max(beta0, 1e-12)
        sentence   = (
            f"The Golden Ratio prior (alpha0={alpha0:.3f}, beta0={beta0:.3f}, "
            f"phi_ratio={phi_ratio:.3f}) contributes {prior_fraction*100:.1f}% "
            f"of the total evidence mass, anchoring the baseline score at "
            f"{prior_mean:.3f}."
        )
        return PriorAttribution(
            alpha0            = round(alpha0, 4),
            beta0             = round(beta0, 4),
            prior_mean        = round(prior_mean, 4),
            phi_ratio         = round(phi_ratio, 4),
            prior_fraction    = round(prior_fraction, 4),
            prior_strength    = round(alpha0 + beta0, 4),
            explanation_sentence = sentence,
        )

    def _build_sentence(
        self,
        modality:    str,
        raw_score:   float,
        confidence:  float,
        norm_attr:   float,
        alignment:   ModalityAlignment,
        contrib_pct: float,
        fused_score: float,
        n_obs:       int,
    ) -> str:
        direction = "raised" if norm_attr > 0 else "lowered"
        contrib_str = (
            f"{contrib_pct:.1f}% of upward attribution"
            if norm_attr > 0 else
            "downward pressure on the score"
        )
        align_str = {
            ModalityAlignment.CONSISTENT:    "consistent with",
            ModalityAlignment.MIXED:         "partially divergent from",
            ModalityAlignment.CONTRADICTING: "contradicting",
        }[alignment]
        obs_str = f"{n_obs} observation{'s' if n_obs != 1 else ''}"

        return (
            f"The '{modality}' modality (score={raw_score:.3f}, "
            f"confidence={confidence:.3f}, {obs_str}) "
            f"{direction} the posterior by {abs(norm_attr):.3f} normalised units, "
            f"contributing {contrib_str}; "
            f"this signal was {align_str} the aggregate score of {fused_score:.3f}."
        )

    def _build_summary(
        self,
        attributions:  List[ModalityAttribution],
        prior_attr:    PriorAttribution,
        fused_score:   float,
        fused_before:  float,
        risk_level:    str,
        dominant:      Optional[str],
    ) -> str:
        delta = fused_score - fused_before
        delta_str = f"{'increased' if delta >= 0 else 'decreased'} by {abs(delta):.4f}"

        dominant_str = (
            f"The dominant signal was '{dominant}'."
            if dominant else
            "No single modality dominated."
        )

        mod_lines = " ".join(m.explanation_sentence for m in attributions)

        prior_str = (
            prior_attr.explanation_sentence
            if self._cfg.include_prior_in_narrative else
            ""
        )

        return (
            f"RISK ASSESSMENT EXPLANATION — Risk level: {risk_level.upper()}. "
            f"The fused posterior score {delta_str} to {fused_score:.4f}. "
            f"{dominant_str} "
            f"{mod_lines} "
            f"{prior_str}"
        ).strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        window: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        **legacy_kwargs: Any,
    ) -> ExplanationReport:
        """
        Generate an ExplanationReport for the current engine state.

        Parameters
        ----------
        window:
            Number of most-recent observations to include in the
            attribution.  0 = use all observations since last reset.
        metadata:
            Optional context (session_id, subject_id, etc.) attached
            to the report.

        Returns
        -------
        ExplanationReport
        """
        # Backward compatibility: older call sites passed snapshot/fused_score/
        # uncertainty/risk_level kwargs.  Keep accepting these arguments and
        # consume only compatible metadata.
        if metadata is None and isinstance(legacy_kwargs.get("metadata"), dict):
            metadata = legacy_kwargs["metadata"]

        engine = self._engine
        records = self._records[-window:] if window > 0 else list(self._records)

        fused_now    = float(getattr(engine, "fused_score", 0.5))
        fused_before = float(records[0].fused_before) if records else fused_now
        alpha0       = float(getattr(engine, "_alpha0", _PHI * 2.0))
        beta0        = float(getattr(engine, "_beta0",  2.0))
        risk_level   = self._classify_risk(fused_now)
        n_total      = len(self._records)

        attributions, prior_attr = self._compute_attribution(
            records   = records,
            alpha0    = alpha0,
            beta0     = beta0,
            fused_now = fused_now,
        )

        dominant = (
            attributions[0].modality
            if attributions and abs(attributions[0].normalized_attribution) > \
               self._cfg.min_contribution_pct / 100.0
            else None
        )

        summary = self._build_summary(
            attributions  = attributions,
            prior_attr    = prior_attr,
            fused_score   = fused_now,
            fused_before  = fused_before,
            risk_level    = risk_level,
            dominant      = dominant,
        )

        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        report = ExplanationReport(
            timestamp             = ts,
            fused_score           = round(fused_now, 6),
            fused_score_before    = round(fused_before, 6),
            risk_level            = risk_level,
            modality_attributions = attributions,
            prior_attribution     = prior_attr,
            dominant_modality     = dominant,
            total_obs_count       = n_total,
            summary_text          = summary,
            metadata              = metadata or {},
        )

        self._reports.append(report)
        if len(self._reports) > self._max_history:
            self._reports.pop(0)

        logger.debug("ExplanationReport generated: %s", report)
        return report

    def report_history(self) -> List[ExplanationReport]:
        """Return all previously generated ExplanationReports."""
        return list(self._reports)

    def reset(self) -> None:
        """Clear observation records and report history."""
        self._records.clear()
        self._reports.clear()

    def __repr__(self) -> str:
        return (
            f"ShapleyExplainer("
            f"records={len(self._records)}, "
            f"reports={len(self._reports)}, "
            f"consistent_thr={self._cfg.consistent_threshold:.3f})"
        )

# ---------------------------------------------------------------------------
# Backward Compatibility Aliases
# ---------------------------------------------------------------------------

ExplainabilityLayer = ShapleyExplainer
XAIConfig            = ExplanationConfig
