"""
Phase2GoldenFusion — Phase 2 Integrated Engine
===============================================
Patent-Pending Technology | NamoNexus Research Team

Integrates Features 2.1, 2.2, and 2.3 into a single engine that
extends TemporalGoldenFusion (Phase 1) with the full Phase 2 stack:

    Phase 1  TemporalGoldenFusion
        ↓ inherits
    Phase2GoldenFusion
        ├── ModalityCalibrator    (2.1) — per-observation confidence adjustment
        ├── SensorTrustScorer     (2.2) — long-term anomaly detection
        └── OnlineHyperparamOptimizer (2.3) — self-tuning config

Observation Pipeline
--------------------
For each call to update(score, confidence, modality_name):

  1. SensorTrustScorer.is_active(modality)
       ↓ False → observation DROPPED
  2. ModalityCalibrator.effective_confidence(...)
       ↓ Returns calibrated confidence
  3. Compute consistency (used for both 2.1 and 2.2)
  4. SensorTrustScorer.record_observation(modality, consistency)
  5. TemporalGoldenFusion.update(score, calibrated_conf, ...)
       ↓ Temporal decay + Bayesian update
  6. Buffer observation for optimizer window
  7. Every opt_interval observations → OnlineHyperparamOptimizer.step(...)
       ↓ May return a new FusionConfig
  8. If improved config found → hot-swap config (preserves current α, β)

New Patent Claims Covered (Phase 2)
-------------------------------------
Claim  8: Modality Auto-Calibration with φ-initialized trust posterior
Claim  9: Long-term sensor trust scoring with Page-Hinkley + φ control limits
Claim 10: Online hyperparameter optimization preserving φ constraint
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config.settings import FusionConfig
from ..core.constants import GOLDEN_RATIO, ENGINE_VERSION
from ..core.exceptions import InvalidObservationError
from ..core.temporal_filter import TemporalConfig
from ..core.temporal_golden_fusion import TemporalGoldenFusion
from ..core.empirical_prior import LearnedPrior
from ..core.modality_calibrator import ModalityCalibrator, CalibrationConfig
from ..core.sensor_trust_scorer import SensorTrustScorer, TrustScorerConfig, TrustEvent
from ..core.hyperopt import OnlineHyperparamOptimizer, HyperparamBounds
from ..core.golden_bayesian import FusionState
from ..utils.validators import validate_score, validate_confidence

logger = logging.getLogger(__name__)

# Observations buffered before each optimizer step
DEFAULT_OPT_INTERVAL = 15


class Phase2GoldenFusion(TemporalGoldenFusion):
    """
    Full Phase 2 fusion engine with auto-calibration, sensor trust scoring,
    and online hyperparameter optimization.

    All Phase 1 capabilities (temporal decay, personalized prior) are
    inherited unchanged.

    Parameters
    ----------
    config:
        Base fusion config (may be updated by optimizer).
    temporal_config:
        Temporal filter config.
    learned_prior:
        Optional personalized prior from EmpiricalPriorLearner.
    calibration_config:
        ModalityCalibrator config.
    trust_config:
        SensorTrustScorer config.
    hyperopt_bounds:
        Search space for online optimizer.
    opt_interval:
        Number of observations between optimizer steps.
    enable_optimizer:
        Set False to disable online optimization (useful for ablation).
    event_callbacks:
        Callbacks for sensor trust events (monitoring integration).

    Examples
    --------
    Basic Phase 2 usage::

        engine = Phase2GoldenFusion()

        engine.update(0.85, 0.70, "text")
        engine.update(0.25, 0.90, "voice")
        engine.update(0.40, 0.60, "face")

        print(engine.fused_score)
        print(engine.calibration_report())
        print(engine.trust_report())

    With event monitoring::

        def on_trust_event(event: TrustEvent):
            print(f"ALERT: {event.event_type} — {event.modality}")

        engine = Phase2GoldenFusion(event_callbacks=[on_trust_event])
    """

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        temporal_config: Optional[TemporalConfig] = None,
        learned_prior: Optional[LearnedPrior] = None,
        calibration_config: Optional[CalibrationConfig] = None,
        trust_config: Optional[TrustScorerConfig] = None,
        trust_scorer_config: Optional[TrustScorerConfig] = None,
        hyperopt_bounds: Optional[HyperparamBounds] = None,
        opt_interval: int = DEFAULT_OPT_INTERVAL,
        enable_optimizer: bool = True,
        event_callbacks: Optional[List[Callable[[TrustEvent], None]]] = None,
    ) -> None:
        super().__init__(
            config=config,
            temporal_config=temporal_config,
            learned_prior=learned_prior,
        )

        # Phase 2 components
        scorer_cfg = trust_config or trust_scorer_config
        self._calibrator = ModalityCalibrator(config=calibration_config)
        self._scorer     = SensorTrustScorer(
            config=scorer_cfg,
            event_callbacks=event_callbacks or [],
        )
        # Backward compatibility aliases used by older tests/scripts.
        self._trust_scorer = self._scorer
        self._optimizer  = OnlineHyperparamOptimizer(bounds=hyperopt_bounds)
        self._opt_interval   = max(1, opt_interval)
        self._enable_optimizer = enable_optimizer

        # Observation buffer for optimizer
        self._opt_buffer_scores:      List[float] = []
        self._opt_buffer_uncertainties: List[float] = []
        self._opt_buffer_risk:         List[str]  = []
        self._opt_buffer_modal:        Dict[str, List[float]] = {}
        self._obs_since_opt:           int = 0

        # Diagnostics
        self._dropped_observations: int = 0
        self._config_swaps:         int = 0

        logger.info(
            "Phase2GoldenFusion v%s | opt_interval=%d optimizer=%s",
            ENGINE_VERSION, opt_interval, enable_optimizer,
        )

    # ------------------------------------------------------------------
    # Overridden update — full Phase 2 pipeline
    # ------------------------------------------------------------------

    def update(
        self,
        score: float,
        confidence: float,
        modality_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Phase2GoldenFusion":
        """
        Full Phase 2 update pipeline.

        Steps (in order):
          1. Validate inputs
          2. Check sensor blacklist → drop if blacklisted
          3. Compute raw consistency (for trust scoring)
          4. Calibrate confidence via ModalityCalibrator
          5. Record observation with SensorTrustScorer
          6. Delegate to TemporalGoldenFusion.update() with calibrated conf
          7. Buffer for optimizer; run optimizer step if interval reached

        Parameters
        ----------
        score, confidence, modality_name, metadata:
            Same as base class.

        Returns
        -------
        Phase2GoldenFusion
            Self.
        """
        strict = self._config.strict_validation
        try:
            score      = validate_score(score, strict=strict)
            confidence = validate_confidence(confidence, strict=strict)
        except ValueError as exc:
            raise InvalidObservationError(str(exc)) from exc

        mod = modality_name or "unknown"

        # ── Step 2: check sensor blacklist ───────────────────────────
        if not self._scorer.is_active(mod):
            self._dropped_observations += 1
            logger.warning(
                "Observation dropped — %s is blacklisted.", mod
            )
            return self

        # ── Step 3: compute consistency ──────────────────────────────
        fused_before = self.fused_score
        consistency  = float(1.0 - abs(score - fused_before))

        # ── Step 4: calibrate confidence ─────────────────────────────
        calibrated_conf = self._calibrator.effective_confidence(
            modality=mod,
            raw_confidence=confidence,
            observation_score=score,
            fused_score_before=fused_before,
        )

        # ── Step 5: record in trust scorer ───────────────────────────
        active = self._scorer.record_observation(mod, consistency)
        if not active:
            # Scorer just blacklisted this sensor mid-update
            self._dropped_observations += 1
            return self

        # ── Step 6: delegate to temporal engine ──────────────────────
        super().update(
            score=score,
            confidence=calibrated_conf,
            modality_name=modality_name,
            metadata={
                **(metadata or {}),
                "raw_confidence":  confidence,
                "calibrated_conf": calibrated_conf,
                "consistency":     consistency,
            },
        )

        # ── Step 7: buffer + optimizer ───────────────────────────────
        self._opt_buffer_scores.append(self.fused_score)
        self._opt_buffer_uncertainties.append(self.uncertainty)
        self._opt_buffer_risk.append(self.risk_level)
        if mod not in self._opt_buffer_modal:
            self._opt_buffer_modal[mod] = []
        self._opt_buffer_modal[mod].append(score)

        self._obs_since_opt += 1

        if self._enable_optimizer and self._obs_since_opt >= self._opt_interval:
            self._run_optimizer_step()
            self._obs_since_opt = 0

        return self

    # ------------------------------------------------------------------
    # Optimizer step (internal)
    # ------------------------------------------------------------------

    def _run_optimizer_step(self) -> None:
        """Run one generation of the online optimizer."""
        new_config = self._optimizer.step(
            current_config=self._config,
            fused_scores=list(self._opt_buffer_scores[-self._opt_interval:]),
            uncertainties=list(self._opt_buffer_uncertainties[-self._opt_interval:]),
            risk_levels=list(self._opt_buffer_risk[-self._opt_interval:]),
            modality_scores={m: list(v[-self._opt_interval:])
                             for m, v in self._opt_buffer_modal.items()},
        )

        if new_config is not None:
            # Hot-swap config — preserves current α, β, temporal state
            old_prior_strength = self._config.prior_strength
            self._config = new_config

            # If prior_strength changed, update prior parameters
            # but preserve accumulated evidence
            if abs(new_config.prior_strength - old_prior_strength) > 0.01:
                alpha_accum = self._alpha - self._alpha0
                beta_accum  = self._beta  - self._beta0
                self._alpha0 = GOLDEN_RATIO * new_config.prior_strength
                self._beta0  = new_config.prior_strength
                self._alpha  = self._alpha0 + alpha_accum
                self._beta   = self._beta0  + beta_accum

            self._config_swaps += 1
            logger.info(
                "Config hot-swapped (swap #%d) | prior_strength=%.3f max_trials=%d",
                self._config_swaps,
                new_config.prior_strength,
                new_config.max_trials,
            )

    # ------------------------------------------------------------------
    # Overridden reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset engine, calibrator, scorer, and optimizer buffers."""
        super().reset()
        self._calibrator.reset()
        self._scorer.reset()
        self._opt_buffer_scores.clear()
        self._opt_buffer_uncertainties.clear()
        self._opt_buffer_risk.clear()
        self._opt_buffer_modal.clear()
        self._obs_since_opt   = 0
        self._dropped_observations = 0

    # ------------------------------------------------------------------
    # Diagnostics and reporting
    # ------------------------------------------------------------------

    def calibration_report(self) -> Dict[str, Any]:
        """Return the modality calibration summary."""
        return self._calibrator.trust_summary()

    def trust_report(self) -> Dict[str, Any]:
        """Return the full sensor trust report."""
        return self._scorer.trust_report()

    def optimizer_report(self) -> Dict[str, Any]:
        """Return optimizer diagnostics."""
        return {
            "n_evaluations":  self._optimizer.n_evaluations,
            "n_generations":  self._optimizer.n_generations
            if hasattr(self._optimizer, "n_generations") else self._optimizer._generation,
            "best_score":     self._optimizer.best_score,
            "config_swaps":   self._config_swaps,
            "current_config": self._config.to_dict(),
        }

    @property
    def dropped_observations(self) -> int:
        """Total number of observations dropped due to blacklisted sensors."""
        return self._dropped_observations

    @property
    def active_modalities(self) -> List[str]:
        """Currently active (non-blacklisted) modalities."""
        return self._scorer.active_modalities

    @property
    def blacklisted_modalities(self) -> List[str]:
        """Currently blacklisted modalities."""
        return self._scorer.blacklisted_modalities

    def add_trust_event_callback(
        self, callback: Callable[[TrustEvent], None]
    ) -> None:
        """Register an additional sensor trust event callback."""
        self._scorer.add_event_callback(callback)

    def force_reinstate_sensor(self, modality: str) -> None:
        """Operator override: manually reinstate a blacklisted sensor."""
        self._scorer.force_reinstate(modality)

    # ------------------------------------------------------------------
    # Extended state
    # ------------------------------------------------------------------

    def get_state(self) -> FusionState:
        """Capture full Phase 2 state."""
        state = super().get_state()
        state.config["phase2"] = {
            "calibration":  self._calibrator.trust_summary(),
            "trust":        self._scorer.trust_report()["sensors"],
            "optimizer":    self._optimizer.get_state().__dict__
            if hasattr(self._optimizer.get_state(), "__dict__") else {},
            "dropped_obs":  self._dropped_observations,
            "config_swaps": self._config_swaps,
        }
        return state

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        active = self._scorer.active_modalities
        bl     = self._scorer.blacklisted_modalities
        return (
            f"Phase2GoldenFusion("
            f"score={self.fused_score:.4f} ± {self.uncertainty:.4f}, "
            f"risk={self.risk_level}, "
            f"active={active}, "
            f"blacklisted={bl}, "
            f"dropped={self._dropped_observations}, "
            f"config_swaps={self._config_swaps})"
        )
