"""
TemporalGoldenFusion — Phase 1 Integrated Engine
=================================================
Patent-Pending Technology | NamoNexus Research Team

This module integrates the two Phase 1 features into a single engine
that is fully backward-compatible with ``GoldenBayesianFusion``:

  Feature 1.1: Temporal Bayesian Filtering  (``TemporalBayesianFilter``)
  Feature 1.2: Empirical Prior Learning     (``EmpiricalPriorLearner``)

Architecture Decision
---------------------
Rather than modifying ``GoldenBayesianFusion`` directly (which would
break the existing API and complicate patent documentation), we use
**composition**:

    TemporalGoldenFusion
        ├── inherits from GoldenBayesianFusion   (all original methods)
        └── composes  TemporalBayesianFilter     (decay logic)

The ``update()`` method is overridden to route evidence through the
temporal filter before committing to α, β.

The ``from_learned_prior()`` class method wires in a personalized prior
from ``EmpiricalPriorLearner``.

New Patent Claims Covered
-------------------------
Claim 6: Temporal Bayesian Filtering with Golden Ratio decay (λ₀ = 1/φ)
Claim 7: Personalized prior via penalized MLE with φ regularization

Backward Compatibility
----------------------
All existing code that uses ``GoldenBayesianFusion`` continues to work
unchanged.  ``TemporalGoldenFusion`` is a drop-in replacement that adds
capabilities when needed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config.settings import FusionConfig
from ..core.constants import GOLDEN_RATIO, GOLDEN_RATIO_RECIPROCAL, ENGINE_VERSION
from ..core.exceptions import InvalidObservationError
from ..core.temporal_filter import TemporalBayesianFilter, TemporalConfig, TemporalSnapshot
from ..core.empirical_prior import (
    EmpiricalPriorLearner,
    LearnedPrior,
    PriorLearningConfig,
    Session,
    SyntheticSessionGenerator,
    SubjectProfile,
)
from ..core.golden_bayesian import GoldenBayesianFusion, FusionState
from ..utils.validators import validate_score, validate_confidence

logger = logging.getLogger(__name__)


class TemporalGoldenFusion(GoldenBayesianFusion):
    """
    Extension of GoldenBayesianFusion with Temporal Bayesian Filtering
    and optional Empirical Prior personalisation.

    All base-class methods (``credible_interval``, ``risk_level``,
    ``deception_probability``, ``get_state``, ``load_state``, etc.)
    are inherited unchanged and work correctly with the temporal state.

    Parameters
    ----------
    config:
        Base fusion configuration (same as GoldenBayesianFusion).
    temporal_config:
        Temporal filter configuration.  Pass ``None`` to use the
        Golden Ratio default (λ = 1/φ).
    learned_prior:
        If provided, overrides the default Golden Ratio prior with a
        personalised prior from ``EmpiricalPriorLearner.fit()``.

    Examples
    --------
    Basic usage with default Golden Ratio decay::

        from namonexus_fusion.phase1 import TemporalGoldenFusion

        engine = TemporalGoldenFusion()
        engine.update(0.85, 0.70, "text")
        engine.update(0.25, 0.90, "voice")
        print(engine.fused_score)

    With personalized prior::

        from namonexus_fusion.phase1 import (
            TemporalGoldenFusion, EmpiricalPriorLearner,
            SyntheticSessionGenerator, SubjectProfile,
        )

        gen = SyntheticSessionGenerator(seed=42)
        profiles, sessions = gen.generate_standard_benchmark()

        learner = EmpiricalPriorLearner()
        learner.add_sessions(sessions)
        prior = learner.fit("subject_calm")

        engine = TemporalGoldenFusion.from_learned_prior(prior)
        engine.update(0.80, 0.70, "text")
        print(f"Personalized prior mean: {prior.prior_mean:.4f}")
        print(f"Fused score: {engine.fused_score:.4f}")

    With adaptive temporal decay::

        from namonexus_fusion.core.temporal_filter import TemporalConfig

        engine = TemporalGoldenFusion(
            temporal_config=TemporalConfig.adaptive(sensitivity=2.0)
        )
    """

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        temporal_config: Optional[TemporalConfig] = None,
        learned_prior: Optional[LearnedPrior] = None,
    ) -> None:
        # If we have a learned prior, create a modified FusionConfig
        # that matches its alpha0/beta0.  We do this by adjusting
        # prior_strength: α₀ = φ · s → s = α₀ / φ.
        if learned_prior is not None:
            # Derive prior_strength from learned alpha0
            inferred_strength = learned_prior.alpha0 / GOLDEN_RATIO
            if config is None:
                from ..config.settings import FusionConfig as FC
                config = FC(prior_strength=inferred_strength)
            else:
                from dataclasses import replace
                config = replace(config, prior_strength=inferred_strength)

            # Override alpha0/beta0 exactly to match the learned prior
            self._learned_prior = learned_prior
        else:
            self._learned_prior = None

        # Initialise the base engine
        super().__init__(config=config)

        # Override alpha0/beta0 if we have a learned prior
        # (base class computes them from prior_strength, which may drift
        # slightly due to rounding; we set exact values here)
        if learned_prior is not None:
            self._alpha0 = learned_prior.alpha0
            self._beta0  = learned_prior.beta0
            self._alpha  = learned_prior.alpha0
            self._beta   = learned_prior.beta0

        # Attach the temporal filter
        t_cfg = temporal_config or TemporalConfig.golden()
        self._temporal = TemporalBayesianFilter(
            alpha0=self._alpha0,
            beta0=self._beta0,
            config=t_cfg,
        )

        logger.info(
            "TemporalGoldenFusion v%s | λ=%.4f adaptive=%s | prior: α₀=%.4f β₀=%.4f "
            "personalized=%s",
            ENGINE_VERSION,
            t_cfg.decay_factor,
            t_cfg.adaptive_decay,
            self._alpha0,
            self._beta0,
            learned_prior is not None,
        )

    # ------------------------------------------------------------------
    # Class method: build from learned prior
    # ------------------------------------------------------------------

    @classmethod
    def from_learned_prior(
        cls,
        prior: LearnedPrior,
        temporal_config: Optional[TemporalConfig] = None,
        fusion_config: Optional[FusionConfig] = None,
    ) -> "TemporalGoldenFusion":
        """
        Construct a TemporalGoldenFusion initialised with a personalised prior.

        Parameters
        ----------
        prior:
            Output of ``EmpiricalPriorLearner.fit()``.
        temporal_config:
            Temporal filter configuration.
        fusion_config:
            Base fusion configuration.

        Returns
        -------
        TemporalGoldenFusion
        """
        return cls(
            config=fusion_config,
            temporal_config=temporal_config,
            learned_prior=prior,
        )

    # ------------------------------------------------------------------
    # Overridden update — routes evidence through temporal filter
    # ------------------------------------------------------------------

    def update(
        self,
        score: float,
        confidence: float,
        modality_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "TemporalGoldenFusion":
        """
        Update the posterior with temporal forgetting applied first.

        Override of ``GoldenBayesianFusion.update()``.

        The temporal filter decays accumulated evidence before adding the
        new observation, ensuring that recent signals carry more weight.

        Parameters
        ----------
        score, confidence, modality_name, metadata:
            Same as base class.

        Returns
        -------
        TemporalGoldenFusion
            Self.
        """
        strict = self._config.strict_validation
        try:
            score      = validate_score(score, strict=strict)
            confidence = validate_confidence(confidence, strict=strict)
        except ValueError as exc:
            raise InvalidObservationError(str(exc)) from exc

        # Compute successes/failures (same as base class)
        n         = self._confidence_to_trials(confidence)
        successes = score * n
        failures  = n - successes

        # ── Temporal filter: decay then update α, β ──────────────────
        current_score = self.fused_score   # score BEFORE this update
        new_alpha, new_beta = self._temporal.apply_decay(
            successes=successes,
            failures=failures,
            current_score=current_score,
            modality=modality_name,
        )

        # Commit decayed + updated values back to the base-class state
        self._alpha = new_alpha
        self._beta  = new_beta
        self._total_observations += n

        # Record in history (inherited)
        entry: Dict[str, Any] = {
            "score":      round(score, 6),
            "confidence": round(confidence, 6),
            "successes":  round(successes, 4),
            "failures":   round(failures, 4),
            "n_trials":   n,
            "modality":   modality_name,
            "lambda":     round(self._temporal.snapshots[-1].lambda_applied, 4),
            "metadata":   metadata,
        }
        self._history.append(entry)
        if len(self._history) > self._config.max_history:
            self._history.pop(0)

        logger.debug(
            "temporal_update | mod=%s score=%.4f λ=%.4f α=%.4f β=%.4f",
            modality_name or "?", score,
            self._temporal.snapshots[-1].lambda_applied,
            self._alpha, self._beta,
        )
        return self

    # ------------------------------------------------------------------
    # Overridden reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset both the base engine and the temporal filter."""
        super().reset()
        self._temporal.reset()

    # ------------------------------------------------------------------
    # Temporal diagnostics
    # ------------------------------------------------------------------

    @property
    def temporal_config(self) -> TemporalConfig:
        """Active temporal filter configuration."""
        return self._temporal.config

    @property
    def effective_lambda(self) -> float:
        """Current effective decay factor."""
        return self._temporal.effective_lambda

    @property
    def effective_observation_count(self) -> float:
        """Effective number of 'fresh' observations in the posterior."""
        return self._temporal.effective_observation_count()

    def lambda_history(self) -> List[float]:
        """λ values from all temporal snapshots."""
        return self._temporal.lambda_history()

    def score_trajectory(self) -> List[Tuple[float, float]]:
        """(timestamp, fused_score) from all temporal snapshots."""
        return self._temporal.score_trajectory()

    @property
    def learned_prior(self) -> Optional[LearnedPrior]:
        """The learned prior used at initialisation, if any."""
        return self._learned_prior

    @property
    def risk_level(self) -> str:
        """
        Backward-compatible risk naming for temporal layer.

        Phase 1 tests and older integrations use ``moderate`` instead of
        ``medium``.
        """
        level = super().risk_level
        return "moderate" if level == "medium" else level

    # ------------------------------------------------------------------
    # Extended state: includes temporal + prior info
    # ------------------------------------------------------------------

    def get_state(self) -> FusionState:
        """Capture state including temporal and prior metadata."""
        state = super().get_state()
        state.config["temporal"] = self._temporal.config.to_dict()
        state.config["learned_prior"] = (
            self._learned_prior.to_dict() if self._learned_prior else None
        )
        return state

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        personalized = (
            f" prior_mean={self._learned_prior.prior_mean:.4f}"
            if self._learned_prior else ""
        )
        return (
            f"TemporalGoldenFusion("
            f"α={self._alpha:.4f}, β={self._beta:.4f}, "
            f"score={self.fused_score:.4f} ± {self.uncertainty:.4f}, "
            f"λ={self.effective_lambda:.4f}, "
            f"eff_obs={self.effective_observation_count:.1f}"
            f"{personalized})"
        )
