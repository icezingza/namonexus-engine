"""
NamoNexusEngine — Unified Facade
=================================
Patent-Pending Technology | NamoNexus Research Team

This module provides the single recommended entry point to the
NamoNexus Fusion Engine.  It wires together all four phases into one
coherent class with a clean, stable public API.

Architecture
------------
NamoNexusEngine is a thin facade over Phase4GoldenFusion.
The underlying inheritance chain is:

    GoldenBayesianFusion          (v3.0 — base posterior)
        └── TemporalGoldenFusion  (Phase 1 — temporal decay + personalised prior)
                └── Phase2GoldenFusion  (Phase 2 — calibration + trust + hyperopt)
                        └── Phase3GoldenFusion  (Phase 3 — drift + streaming)
                                └── Phase4GoldenFusion  (Phase 4 — XAI + hierarchical)
                                        └── NamoNexusEngine  (this facade)

All Phase 1–4 functionality is available directly on this object.

Quick Start
-----------
    from namonexus_fusion import NamoNexusEngine

    engine = NamoNexusEngine()
    engine.update(0.85, 0.70, "text")
    engine.update(0.25, 0.90, "voice")
    engine.update(0.60, 0.85, "face")

    print(engine.fused_score)       # Bayesian posterior mean
    print(engine.risk_level)         # "low" | "medium" | "high" | "critical"
    report = engine.explain()       # GDPR/PDPA Shapley attribution
    print(report.summary_text)

Hierarchical / Federated Learning
-----------------------------------
    from namonexus_fusion import NamoNexusEngine
    from namonexus_fusion.core.hierarchical_bayesian import HierarchicalBayesianModel

    model  = HierarchicalBayesianModel()
    engine = NamoNexusEngine(hierarchical_model=model, subject_id="user_001")

    engine.update(0.8, 0.9, "text")
    engine.commit_session()

    delta = model.export_federated_delta()
    # forward delta to federated coordinator (no raw data transmitted)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .core.phase4_fusion import Phase4GoldenFusion
from .core.constants import ENGINE_VERSION
from .config.settings import FusionConfig

logger = logging.getLogger(__name__)


class NamoNexusEngine(Phase4GoldenFusion):
    """
    NamoNexus Fusion Engine — complete multimodal Bayesian fusion stack.

    Provides all capabilities from Phases 1 through 4 via a single,
    stable class.  Use this class for all production deployments.

    Parameters
    ----------
    config : FusionConfig, optional
        Bayesian fusion configuration.  Defaults to Golden Ratio prior
        with prior_strength=1.0.
    temporal_config : TemporalConfig, optional
        Temporal decay configuration.  Default: λ = 1/φ.
    learned_prior : LearnedPrior, optional
        Personalised Beta prior from EmpiricalPriorLearner.
    calibration_config : CalibrationConfig, optional
        Modality auto-calibration settings.
    trust_config : TrustScorerConfig, optional
        Sensor trust scoring settings.
    hyperopt_bounds : HyperparamBounds, optional
        Search space for online hyperparameter optimiser.
    opt_interval : int
        Number of observations between optimiser steps. Default: 15.
    enable_optimizer : bool
        Enable/disable the online hyperparameter optimiser. Default: True.
    drift_config : DriftConfig, optional
        Drift detector configuration.  Default: h=φ², δ=1/φ².
    streaming_config : StreamingConfig, optional
        Streaming pipeline configuration.
    xai_config : XAIConfig, optional
        Explainability layer configuration.
    hierarchical_model : HierarchicalBayesianModel, optional
        Attach a hierarchical model for federated learning support.
    subject_id : str, optional
        Subject identifier.  Required when hierarchical_model is provided.

    Examples
    --------
    Basic multimodal fusion::

        engine = NamoNexusEngine()
        engine.update(0.85, 0.70, "text")
        engine.update(0.25, 0.90, "voice")
        engine.update(0.60, 0.85, "face")
        print(engine)

    With all Phase 2 options::

        engine = NamoNexusEngine(
            config=FusionConfig(prior_strength=2.0),
            enable_optimizer=False,
        )

    With streaming::

        from namonexus_fusion.core.streaming_pipeline import (
            StreamingObservation, InMemoryConnector
        )
        pipeline = engine.stream()
        obs = [
            StreamingObservation(score=0.8, confidence=0.9, modality="text"),
            StreamingObservation(score=0.3, confidence=0.7, modality="voice"),
        ]
        results = pipeline.run_sync(InMemoryConnector(obs))

    XAI explanation::

        report = engine.explain(metadata={"session_id": "sess_001"})
        audit  = report.to_dict()          # JSON-serialisable
    """

    VERSION = ENGINE_VERSION

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        logger.info(
            "NamoNexusEngine %s initialised | "
            "φ=%.6f | prior α₀=%.4f β₀=%.4f",
            self.VERSION,
            (1.0 + 5.0 ** 0.5) / 2.0,
            getattr(self, "_alpha0", 0.0),
            getattr(self, "_beta0",  0.0),
        )

    # ------------------------------------------------------------------
    # Convenience: multi-modal batch update
    # ------------------------------------------------------------------

    def update_batch(
        self,
        observations: List[Dict[str, Any]],
    ) -> "NamoNexusEngine":
        """
        Apply multiple observations in sequence.

        Parameters
        ----------
        observations
            List of dicts, each with keys:
            ``score`` (float), ``confidence`` (float),
            ``modality`` (str, optional), ``metadata`` (dict, optional).

        Returns
        -------
        NamoNexusEngine
            Self (fluent interface).

        Example
        -------
        ::

            engine.update_batch([
                {"score": 0.85, "confidence": 0.70, "modality": "text"},
                {"score": 0.25, "confidence": 0.90, "modality": "voice"},
                {"score": 0.60, "confidence": 0.85, "modality": "face"},
            ])
        """
        failed_indices: List[int] = []
        for idx, obs in enumerate(observations):
            try:
                self.update(
                    score         = float(obs["score"]),
                    confidence    = float(obs["confidence"]),
                    modality_name = obs.get("modality") or obs.get("modality_name"),
                    metadata      = obs.get("metadata"),
                )
            except Exception as exc:
                logger.error("Observation at index %d failed: %s", idx, exc)
                failed_indices.append(idx)
        
        if failed_indices:
            logger.warning(
                "Batch update completed with %d failures at indices: %s",
                len(failed_indices), failed_indices
            )
        
        return self

    # ------------------------------------------------------------------
    # Convenience: full session summary
    # ------------------------------------------------------------------

    def session_summary(self) -> Dict[str, Any]:
        """
        Return a structured summary of the current session.

        Includes fused score, risk level, uncertainty, credible interval,
        calibration report, trust report, drift summary, and XAI
        explanation if any modality observations have been recorded.

        Returns
        -------
        dict
            JSON-serialisable session summary.
        """
        lo, hi = self.credible_interval()

        summary: Dict[str, Any] = {
            "version":          self.VERSION,
            "fused_score":      round(self.fused_score, 6),
            "risk_level":       self.risk_level,
            "uncertainty":      round(self.uncertainty, 6),
            "credible_interval": {"lower": round(lo, 4), "upper": round(hi, 4)},
            "total_observations": self.total_observations,
        }

        # Phase 2: calibration + trust
        if hasattr(self, "calibration_report"):
            summary["calibration"] = self.calibration_report()
        if hasattr(self, "trust_report"):
            summary["trust"]       = self.trust_report()

        # Phase 3: drift
        if hasattr(self, "drift_summary"):
            summary["drift"] = self.drift_summary()

        # Phase 4: XAI
        if hasattr(self, "xai") and getattr(self.xai, "_records", None):
            report = self.explain()
            summary["explanation"] = {
                "summary_text":        report.summary_text,
                "dominant_modality":   report.dominant_modality,
                "modality_count":      len(report.modality_attributions),
                
                
            }

        # Hierarchical
        if getattr(self, "subject_id", None):
            summary["subject_id"] = self.subject_id

        return summary

    def commit_session(self) -> Optional[Any]:
        """Forward commit to Phase 4 compatibility layer."""
        return super().commit_session()  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        try:
            lo, hi = self.credible_interval()
            return (
                f"NamoNexusEngine("
                f"v{self.VERSION} | "
                f"score={self.fused_score:.4f} ± {self.uncertainty:.4f} | "
                f"CI=[{lo:.3f},{hi:.3f}] | "
                f"risk={self.risk_level} | "
                f"obs={self.total_observations:.0f})"
            )
        except Exception:
            return f"NamoNexusEngine(v{self.VERSION})"
