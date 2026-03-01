# © 2026 Kanin Raksaraj (P'Ice). All Rights Reserved.
# Confidential and Proprietary. Do not distribute without permission.

"""
phase4_fusion.py — Phase 4 Integrated Engine: Commercialization Layer
======================================================================
Patent-Pending Technology | NamoNexus Research Team

Integrates Features 4.1 and 4.2 into a single commercial-grade engine
that extends Phase3GoldenFusion (which itself extends Phases 1 + 2):

    Phase 1  TemporalGoldenFusion
        ↓ inherits
    Phase 2  Phase2GoldenFusion
        ↓ inherits
    Phase 3  Phase3GoldenFusion
        ↓ inherits
    Phase 4  Phase4GoldenFusion
        ├── ShapleyExplainer        (4.1) — XAI attribution + human-readable output
        └── HierarchicalBayesian    (4.2) — Population/Individual prior + Federated

Architecture Decision
---------------------
Phase 4 is the **commercialization layer** — it adds:

1. Automatic explainability on every inference:
   ``engine.explain()`` returns a full ``ExplanationReport`` with Shapley
   attributions weighted by Golden Ratio structural prior (Patent Claim 13).

2. Hierarchical prior management:
   ``engine.local_model`` holds a ``LocalModel`` that is warm-started from
   a shared ``PopulationModel``.  The local model's inferences feed the
   standard Beta-posterior engine.

3. Federated Learning integration:
   ``engine.contribute_to_federation(aggregator)`` registers this engine's
   local model into a ``FederatedAggregator`` and (optionally) triggers
   an aggregation round.

New Patent Claims Covered (Phase 4)
-------------------------------------
Claim 13: Shapley-value attribution weighted by Golden Ratio structural prior
          → human-readable per-modality explanation (Feature 4.1)
Claim 14: Hierarchical Bayesian architecture with population/individual split,
          φ-constrained at all levels, supporting Federated Learning (Feature 4.2)

Usage
-----
::

    # ── Standalone explainability ─────────────────────────────────
    engine = Phase4GoldenFusion()
    engine.update(0.85, 0.90, "text")
    engine.update(0.60, 0.70, "voice")
    engine.update(0.40, 0.80, "face")

    report = engine.explain()
    print(report.narrative)
    print(report.to_audit_dict())   # PDPA / GDPR / FDA ready

    # ── Hierarchical: cold-start from population ──────────────────
    pop    = PopulationModel()
    engine = Phase4GoldenFusion(population=pop)
    engine.update(0.75, 0.80, "text")
    print(engine.local_model.fused_score)

    # ── Federated: share stats, update population ─────────────────
    aggregator = FederatedAggregator(pop)
    engine.contribute_to_federation(aggregator)
    aggregator.aggregate()           # updates pop in-place

    # New engine cold-starts from updated population
    engine2 = Phase4GoldenFusion(population=pop)
    print(engine2.local_model.fused_score)  # benefiting from federated knowledge
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .explainability import (
    ExplanationConfig,
    ExplanationReport,
    ShapleyExplainer,
)
from .hierarchical_bayes import (
    FederatedAggregator,
    HierarchicalConfig,
    LocalModel,
    PopulationModel,
)
try:
    from .hierarchical_bayesian import (
        HierarchicalBayesianModel,
        BlendedPrior,
        IndividualPosterior,
    )
except Exception:  # pragma: no cover - optional compatibility path
    HierarchicalBayesianModel = Any  # type: ignore
    BlendedPrior = Any               # type: ignore
    IndividualPosterior = Any        # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import Phase 3 as base; fall back to a minimal stub
# ---------------------------------------------------------------------------

try:
    from .phase3_fusion import Phase3GoldenFusion  # type: ignore
    _BASE_CLASS = Phase3GoldenFusion
    _BASE_NAME  = "Phase3GoldenFusion"
except ImportError:
    logger.warning(
        "Phase3GoldenFusion not found — Phase4GoldenFusion will use "
        "_MinimalStub as base class.  Place phase3/ on PYTHONPATH for production."
    )

    # --- Minimal stub mirroring Phase 3 interface ---
    class _MinimalStub:
        """
        Minimal stub for standalone Phase 4 testing.
        Simulates a Beta-posterior fusion engine with temporal decay and
        drift/streaming stubs.
        """

        _PHI: float = (1.0 + math.sqrt(5.0)) / 2.0

        def __init__(self, **kwargs: Any) -> None:
            self._alpha    = self._PHI * 2.0
            self._beta     = 2.0
            self._alpha0   = self._alpha
            self._beta0    = self._beta
            self._history: List[Dict[str, Any]] = []

        def update(
            self,
            score:         float,
            confidence:    float,
            modality_name: Optional[str] = None,
            metadata:      Optional[Dict[str, Any]] = None,
            **kwargs: Any,
        ) -> "_MinimalStub":
            score      = float(np.clip(score,      0.0, 1.0))
            confidence = float(np.clip(confidence, 0.0, 1.0))
            n          = max(1.0, confidence * 10.0)
            self._alpha += score * n
            self._beta  += (1.0 - score) * n
            self._history.append({
                "modality":   modality_name or "unknown",
                "score":      score,
                "confidence": confidence,
            })
            return self

        @property
        def fused_score(self) -> float:
            return self._alpha / (self._alpha + self._beta)

        @property
        def uncertainty(self) -> float:
            a, b = self._alpha, self._beta
            n    = a + b
            return math.sqrt(a * b / (n ** 2 * (n + 1))) if n > 1 else 0.5

        @property
        def risk_level(self) -> str:
            s = self.fused_score
            if s >= 0.75: return "low"
            if s >= 0.45: return "medium"
            return "high"

        def get_state(self) -> Dict[str, Any]:
            return {
                "alpha":            self._alpha,
                "beta":             self._beta,
                "modality_history": list(self._history),
                "config":           {},
            }

        def reset(self) -> None:
            self._alpha = self._alpha0
            self._beta  = self._beta0
            self._history.clear()

        # Stub drift / stream interface (Phase 3 API surface)
        @property
        def has_drift_alarm(self) -> bool:
            return False

        def drift_events(self, **kwargs: Any) -> List:
            return []

        def drift_summary(self) -> Dict[str, Any]:
            return {}

    _BASE_CLASS = _MinimalStub   # type: ignore
    _BASE_NAME  = "_MinimalStub"


# ---------------------------------------------------------------------------
# Phase 4 integrated engine
# ---------------------------------------------------------------------------


class Phase4GoldenFusion(_BASE_CLASS):
    """
    Commercial-grade fusion engine with Explainability + Hierarchical Bayesian.

    Extends Phase3GoldenFusion (Phase 1 + 2 + 3) with:

    Feature 4.1 — ShapleyExplainer:
        Provides ``engine.explain()`` returning an ``ExplanationReport`` with:
          • Per-modality Shapley value (raw + φ-weighted)
          • Human-readable narrative per modality
          • Compliance audit dict (PDPA / GDPR / FDA)

    Feature 4.2 — HierarchicalBayesian:
        Maintains a ``LocalModel`` warm-started from a shared ``PopulationModel``.
        ``engine.contribute_to_federation(aggregator)`` shares only sufficient
        statistics (α, β) — no raw observations — for federated updates.

    Parameters
    ----------
    explanation_config:
        ``ExplanationConfig`` for the ShapleyExplainer.  Defaults to
        ``ExplanationConfig.default()``.
    population:
        Shared ``PopulationModel``.  If None, a new standalone model is created.
    hierarchical_config:
        ``HierarchicalConfig``.  Used when creating the local model.
    client_id:
        Identifier for this engine in federated settings.
    explanation_language:
        "en" or "th" — overrides config if provided.
    auto_explain_on_update:
        If True, ``explain()`` is called after every ``update()`` and the
        last report is stored in ``self.last_report``.
    **base_kwargs:
        Forwarded to Phase3GoldenFusion (or stub).
    """

    def __init__(
        self,
        explanation_config:     Optional[ExplanationConfig]    = None,
        xai_config:             Optional[ExplanationConfig]    = None,
        population:             Optional[PopulationModel]      = None,
        hierarchical_config:    Optional[HierarchicalConfig]   = None,
        client_id:              Optional[str]                  = None,
        hierarchical_model:     Optional[Any]                  = None,
        subject_id:             Optional[str]                  = None,
        explanation_language:   Optional[str]                  = None,
        auto_explain_on_update: bool                           = False,
        **base_kwargs: Any,
    ) -> None:
        # Backward compatibility: accept legacy xai_config alias.
        if explanation_config is None:
            explanation_config = xai_config or base_kwargs.pop("xai_config", None)

        super().__init__(**base_kwargs)

        # 4.1 — Explainability
        exp_cfg = explanation_config or ExplanationConfig.default()
        lang = explanation_language or "en"
        self._explainer            = ShapleyExplainer(engine=self, config=exp_cfg)
        self._auto_explain         = auto_explain_on_update
        self.last_report: Optional[ExplanationReport] = None

        # 4.2 — Hierarchical Bayesian
        hier_cfg        = hierarchical_config or HierarchicalConfig.default()
        self._population = population or PopulationModel(config=hier_cfg)
        self.local_model = LocalModel.from_population(
            self._population,
            client_id=client_id,
            config=hier_cfg,
        )

        logger.info(
            "Phase4GoldenFusion | base=%s client=%s pop_mean=%.4f lang=%s auto_explain=%s",
            _BASE_NAME,
            self.local_model.client_id,
            self._population.mean,
            lang,
            auto_explain_on_update,
        )

        # 4.2 compatibility path (legacy hierarchical_bayesian module)
        self._hierarchical_model = hierarchical_model
        self._subject_id = subject_id
        self._last_commit_alpha = float(getattr(self, "_alpha", 0.0))
        self._last_commit_beta = float(getattr(self, "_beta", 0.0))
        self._last_commit_obs = float(getattr(self, "total_observations", 0.0))
        if self._hierarchical_model is not None and self._subject_id:
            self._warm_start_from_hierarchical()

    def _warm_start_from_hierarchical(self) -> None:
        """
        Optional warm start from legacy HierarchicalBayesianModel.

        This keeps Phase 4 compatible with older public APIs that pass
        ``hierarchical_model`` + ``subject_id`` to the constructor.
        """
        if self._hierarchical_model is None or not self._subject_id:
            return
        try:
            blended = self._hierarchical_model.get_blended_prior(self._subject_id)
            alpha_prior = float(getattr(blended, "alpha_blend"))
            beta_prior = float(getattr(blended, "beta_blend"))
        except Exception as exc:
            logger.warning("Hierarchical warm-start skipped: %s", exc)
            return

        if alpha_prior <= 0 or beta_prior <= 0:
            return

        self._alpha0 = alpha_prior
        self._beta0 = beta_prior
        self._alpha = alpha_prior
        self._beta = beta_prior

        # Keep temporal filter internals coherent with the overridden prior.
        temporal = getattr(self, "_temporal", None)
        if temporal is not None:
            temporal._alpha0 = alpha_prior  # type: ignore[attr-defined]
            temporal._beta0 = beta_prior    # type: ignore[attr-defined]
            temporal.reset()

        self._last_commit_alpha = alpha_prior
        self._last_commit_beta = beta_prior
        self._last_commit_obs = float(getattr(self, "total_observations", 0.0))

    # ------------------------------------------------------------------
    # Overridden update — syncs LocalModel in parallel
    # ------------------------------------------------------------------

    def update(
        self,
        score:         float,
        confidence:    float,
        modality_name: Optional[str] = None,
        metadata:      Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "Phase4GoldenFusion":
        """
        Update base engine posterior AND individual LocalModel.

        Both the Phase 1-3 Beta-posterior chain and the hierarchical
        LocalModel receive the same observation.

        Parameters
        ----------
        score:
            Modality score ∈ [0, 1].
        confidence:
            Reliability ∈ [0, 1].
        modality_name:
            Sensing modality name.
        metadata:
            Optional key-value metadata (stored in history for XAI).

        Returns
        -------
        Phase4GoldenFusion
            Self (fluent interface).
        """
        modality = modality_name or "unknown"
        fused_before = float(getattr(self, "fused_score", 0.5))
        alpha_before = float(getattr(self, "_alpha", 0.0))
        beta_before = float(getattr(self, "_beta", 0.0))

        # Update Phase 1-3 engine
        super().update(
            score         = score,
            confidence    = confidence,
            modality_name = modality_name,
            metadata      = metadata,
            **kwargs,
        )

        fused_after = float(getattr(self, "fused_score", fused_before))
        alpha_after = float(getattr(self, "_alpha", alpha_before))
        beta_after = float(getattr(self, "_beta", beta_before))

        # Capture explainability record for this observation.
        self._explainer.record(
            modality=modality,
            raw_score=float(score),
            raw_confidence=float(confidence),
            trust_factor=1.0,
            fused_before=fused_before,
            fused_after=fused_after,
            alpha_before=alpha_before,
            beta_before=beta_before,
            alpha_after=alpha_after,
            beta_after=beta_after,
        )

        # Sync hierarchical local model
        self.local_model.update(
            score      = score,
            confidence = confidence,
            modality   = modality,
            metadata   = metadata,
        )

        # Optional: auto-generate explanation after each update
        if self._auto_explain:
            self.last_report = self.explain()

        return self

    # ------------------------------------------------------------------
    # Feature 4.1 — Explainability
    # ------------------------------------------------------------------

    def explain(
        self,
        window: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        explanation_config: Optional[ExplanationConfig] = None,
    ) -> ExplanationReport:
        """
        Generate a ``ExplanationReport`` for the current inference state.

        Uses Shapley-value attribution weighted by Golden Ratio structural
        prior contribution (Patent Claim 13).

        Parameters
        ----------
        explanation_config:
            Override engine-level config for this call only.

        Returns
        -------
        ExplanationReport
            Full attribution report with narrative and audit dict.
        """
        explainer = self._explainer
        if explanation_config is not None:
            # One-off override without mutating the engine-level explainer.
            explainer = ShapleyExplainer(engine=self, config=explanation_config)
            explainer._records = list(self._explainer._records)  # type: ignore[attr-defined]

        report = explainer.explain(window=window, metadata=metadata)
        self.last_report = report
        return report

    def explain_audit(self) -> Dict[str, Any]:
        """
        Convenience: explain and return compliance-ready audit dict immediately.

        Suitable for direct insertion into PDPA / GDPR / FDA audit logs.
        """
        return self.explain().to_audit_dict()

    # ------------------------------------------------------------------
    # Feature 4.2 — Federated Learning
    # ------------------------------------------------------------------

    def contribute_to_federation(
        self,
        aggregator:  FederatedAggregator,
        auto_aggregate: bool = False,
    ) -> None:
        """
        Register this engine's ``LocalModel`` as a federated participant.

        Parameters
        ----------
        aggregator:
            The ``FederatedAggregator`` managing the shared population.
        auto_aggregate:
            If True, immediately trigger one aggregation round after registration.
            Useful in synchronous federated settings.
        """
        aggregator.register(self.local_model)
        logger.info(
            "Phase4GoldenFusion: client %s registered in FederatedAggregator "
            "(auto_aggregate=%s)",
            self.local_model.client_id, auto_aggregate,
        )
        if auto_aggregate:
            aggregator.aggregate()

    def refresh_from_population(self) -> None:
        """
        Soft-pull the local model toward the updated population prior.

        Call this after receiving an aggregated ``PopulationModel`` update
        to incorporate collective knowledge without losing local observations.

        Implementation: a one-step Bayesian update that adds (α_pop_delta, β_pop_delta)
        to the local posterior, scaled by ``HierarchicalConfig.individual_scale_multiplier``.
        """
        pop_mean  = self._population.mean
        pop_conc  = self._population.concentration
        scale     = self.local_model._cfg.individual_scale_multiplier
        alpha_add = pop_mean * pop_conc * scale
        beta_add  = (1.0 - pop_mean) * pop_conc * scale

        # Directly adjust local model posterior (without adding to history)
        self.local_model._alpha += alpha_add
        self.local_model._beta  += beta_add
        logger.debug(
            "refresh_from_population | client=%s Δα=%.4f Δβ=%.4f",
            self.local_model.client_id, alpha_add, beta_add,
        )

    # ------------------------------------------------------------------
    # Population accessors
    # ------------------------------------------------------------------

    @property
    def population(self) -> PopulationModel:
        """The shared population model."""
        return self._population

    @property
    def population_mean(self) -> float:
        """Current population-level belief (useful for cold-start diagnostics)."""
        return self._population.mean

    @property
    def hierarchical_gap(self) -> float:
        """
        |individual_score − population_mean|.

        Large gap suggests the individual deviates strongly from the population.
        """
        return abs(self.fused_score - self._population.mean)

    # ------------------------------------------------------------------
    # Overridden reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset base engine + local model (population prior preserved)."""
        super().reset()
        self.local_model.reset()
        self._explainer.reset()
        self.last_report = None
        self._last_commit_alpha = float(getattr(self, "_alpha", 0.0))
        self._last_commit_beta = float(getattr(self, "_beta", 0.0))
        self._last_commit_obs = float(getattr(self, "total_observations", 0.0))

    @property
    def xai(self) -> ShapleyExplainer:
        """Compatibility alias used by tests/examples."""
        return self._explainer

    @property
    def hierarchical_model(self) -> Optional[Any]:
        """Return the attached legacy HierarchicalBayesianModel if any."""
        return self._hierarchical_model

    @property
    def subject_id(self) -> Optional[str]:
        return self._subject_id

    def blended_prior(self) -> Optional[Any]:
        """Return blended prior from legacy hierarchical model when available."""
        if self._hierarchical_model is None or not self._subject_id:
            return None
        try:
            return self._hierarchical_model.get_blended_prior(self._subject_id)
        except Exception as exc:
            logger.warning("blended_prior unavailable: %s", exc)
            return None

    def commit_session(self) -> Optional[Any]:
        """
        Commit newly accumulated evidence to legacy HierarchicalBayesianModel.
        """
        if self._hierarchical_model is None or not self._subject_id:
            return None

        alpha_now = float(getattr(self, "_alpha", 0.0))
        beta_now = float(getattr(self, "_beta", 0.0))
        obs_now = float(getattr(self, "total_observations", 0.0))

        alpha_delta = max(0.0, alpha_now - self._last_commit_alpha)
        beta_delta = max(0.0, beta_now - self._last_commit_beta)
        n_obs_delta = max(0, int(round(obs_now - self._last_commit_obs)))

        posterior = self._hierarchical_model.update_individual(
            self._subject_id,
            alpha_delta=alpha_delta,
            beta_delta=beta_delta,
            n_obs=n_obs_delta,
            n_sessions=1,
        )

        self._last_commit_alpha = alpha_now
        self._last_commit_beta = beta_now
        self._last_commit_obs = obs_now
        return posterior

    def full_reset(self) -> None:
        """Reset everything including the population model."""
        self.reset()
        self._population.reset()

    # ------------------------------------------------------------------
    # Extended state
    # ------------------------------------------------------------------

    def get_state(self) -> Any:
        """Full state including Phase 4 metadata."""
        state = super().get_state()

        local_state = self.local_model.get_state()

        if isinstance(state, dict):
            state["phase4"] = {
                "explainability": {
                    "last_report": self.last_report.to_audit_dict()
                    if self.last_report else None,
                },
                "hierarchical": {
                    "client_id":      local_state["client_id"],
                    "local_score":    local_state["fused_score"],
                    "population_mean": self._population.mean,
                    "hierarchical_gap": self.hierarchical_gap,
                    "population_agg_count": self._population.aggregation_count,
                },
            }
            # Ensure modality_history is available at top level for explainer
            if "modality_history" not in state:
                state["modality_history"] = local_state.get("modality_history", [])
        return state

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        score   = getattr(self, "fused_score",  0.0)
        uncert  = getattr(self, "uncertainty",  0.0)
        risk_val = getattr(self, "risk_level",   "unknown")
        if callable(risk_val):
            risk_val = risk_val()
        alarms  = getattr(self, "drift_alarm_count", 0)
        pop_m   = self._population.mean
        return (
            f"Phase4GoldenFusion("
            f"score={score:.4f} ± {uncert:.4f}, "
            f"risk={risk_val}, "
            f"drift_alarms={alarms}, "
            f"xai_records={len(self._explainer._records)}, "
            f"client={self.local_model.client_id}, "
            f"pop_mean={pop_m:.4f}, "
            f"gap={self.hierarchical_gap:.4f})"
        )
