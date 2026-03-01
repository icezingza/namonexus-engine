"""
GoldenBayesianFusion — Core Bayesian Multimodal Fusion Engine (v3.0)
====================================================================
Patent-Pending Technology | NamoNexus Research Team

Foundation engine using a Beta-Binomial model anchored on the Golden Ratio:

    Prior: α₀ = φ · s,  β₀ = s       (so α₀/β₀ = φ)

Observation update:
    n       = confidence × confidence_scale   (pseudo-trial count)
    success = score × n
    failure = n − success
    α ← α + success
    β ← β + failure

Fused score = α / (α + β)  ∈ (0, 1)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta as beta_dist  # type: ignore

from ..config.settings import FusionConfig
from ..core.constants import GOLDEN_RATIO, RISK_LOW_THRESHOLD, RISK_MEDIUM_THRESHOLD, RISK_HIGH_THRESHOLD
from ..core.exceptions import InvalidObservationError
from ..utils.validators import validate_score, validate_confidence

logger = logging.getLogger(__name__)


@dataclass
class FusionState:
    """Serialisable snapshot of engine state for persistence / transfer."""
    alpha: float
    beta: float
    total_observations: float
    history: List[Dict[str, Any]]
    config: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FusionState":
        return cls(**d)


class GoldenBayesianFusion:
    """
    Core Bayesian fusion engine with Golden Ratio prior.

    Thread safety: not thread-safe by default.  Use external locking
    or create one engine per thread.
    """

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self._config: FusionConfig = config or FusionConfig()
        self._alpha0: float = self._config.alpha0
        self._beta0: float = self._config.beta0
        self._alpha: float = self._alpha0
        self._beta: float = self._beta0
        self._total_observations: float = 0.0
        self._history: List[Dict[str, Any]] = []

        logger.info(
            "GoldenBayesianFusion | α₀=%.4f β₀=%.4f (ratio=%.4f, φ=%.4f)",
            self._alpha0, self._beta0,
            self._alpha0 / self._beta0, GOLDEN_RATIO,
        )

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def fused_score(self) -> float:
        """Posterior mean of the Beta distribution: α / (α + β)."""
        return self._alpha / (self._alpha + self._beta)

    @property
    def uncertainty(self) -> float:
        """
        Posterior standard deviation of the Beta distribution.

        σ = sqrt(αβ / ((α+β)² (α+β+1)))
        """
        a, b = self._alpha, self._beta
        n = a + b
        return float(np.sqrt(a * b / (n * n * (n + 1))))

    @property
    def total_observations(self) -> float:
        return self._total_observations

    @property
    def alpha0(self) -> float:
        """Prior alpha parameter (legacy compatibility)."""
        return self._alpha0

    @property
    def beta0(self) -> float:
        """Prior beta parameter (legacy compatibility)."""
        return self._beta0

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    # ── Core update ─────────────────────────────────────────────────────────

    def _confidence_to_trials(self, confidence: float) -> float:
        """Convert a confidence value to pseudo-trial count."""
        n = confidence * self._config.confidence_scale
        return float(np.clip(n, 0.0, self._config.max_trials))

    def update(
        self,
        score: float,
        confidence: float,
        modality_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "GoldenBayesianFusion":
        """
        Incorporate a new observation.

        Parameters
        ----------
        score:       Observable outcome in [0, 1].
        confidence:  Reliability of this observation in [0, 1].
        modality_name: Label for logging (e.g. "face", "voice", "text").
        metadata:    Arbitrary extra data recorded in history.

        Returns
        -------
        Self (for chaining).
        """
        strict = self._config.strict_validation
        try:
            score = validate_score(score, strict=strict)
            confidence = validate_confidence(confidence, strict=strict)
        except ValueError as exc:
            raise InvalidObservationError(str(exc)) from exc

        n = self._confidence_to_trials(confidence)
        successes = score * n
        failures = n - successes

        self._alpha += successes
        self._beta += failures
        self._total_observations += n

        entry: Dict[str, Any] = {
            "score": round(score, 6),
            "confidence": round(confidence, 6),
            "successes": round(successes, 4),
            "failures": round(failures, 4),
            "n_trials": n,
            "modality": modality_name,
            "metadata": metadata,
        }
        self._history.append(entry)
        if len(self._history) > self._config.max_history:
            self._history.pop(0)

        logger.debug(
            "update | mod=%s score=%.4f conf=%.4f → α=%.4f β=%.4f fused=%.4f",
            modality_name or "?", score, confidence,
            self._alpha, self._beta, self.fused_score,
        )
        return self

    # ── Risk & interpretation ────────────────────────────────────────────────

    @property
    def risk_level(self) -> str:
        """
        Map fused_score to a categorical risk level.

        Returns
        -------
        "low" | "medium" | "high" | "critical"
        """
        s = self.fused_score
        if s < RISK_LOW_THRESHOLD:
            return "low"
        elif s < RISK_MEDIUM_THRESHOLD:
            return "medium"
        elif s < RISK_HIGH_THRESHOLD:
            return "high"
        else:
            return "critical"

    def deception_probability(
        self,
        threshold: float = 0.5,
        confidence: Optional[float] = None,
    ) -> float:
        """P(score > threshold) under the current Beta posterior."""
        # ``confidence`` is accepted for backward compatibility with older APIs.
        _ = confidence
        return float(1.0 - beta_dist.cdf(threshold, self._alpha, self._beta))

    def credible_interval(self, credibility: float = 0.95) -> Tuple[float, float]:
        """
        Bayesian credible interval for the fused score.

        Parameters
        ----------
        credibility: Width of the interval (default 0.95 = 95%).

        Returns
        -------
        (lower, upper) bounds.
        """
        alpha_ci = (1.0 - credibility) / 2.0
        lo = float(beta_dist.ppf(alpha_ci, self._alpha, self._beta))
        hi = float(beta_dist.ppf(1.0 - alpha_ci, self._alpha, self._beta))
        return lo, hi

    # ── State persistence ────────────────────────────────────────────────────

    def get_state(self) -> FusionState:
        """Capture full engine state for serialisation."""
        return FusionState(
            alpha=self._alpha,
            beta=self._beta,
            total_observations=self._total_observations,
            history=list(self._history),
            config=self._config.to_dict(),
        )

    def load_state(self, state: FusionState) -> None:
        """Restore engine from a previously captured state."""
        self._alpha = state.alpha
        self._beta = state.beta
        self._total_observations = state.total_observations
        self._history = list(state.history)

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset posterior to prior."""
        self._alpha = self._alpha0
        self._beta = self._beta0
        self._total_observations = 0.0
        self._history.clear()
        logger.debug("GoldenBayesianFusion reset to prior.")

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        lo, hi = self.credible_interval()
        return (
            f"GoldenBayesianFusion("
            f"α={self._alpha:.4f}, β={self._beta:.4f}, "
            f"score={self.fused_score:.4f} ± {self.uncertainty:.4f}, "
            f"CI=[{lo:.3f},{hi:.3f}], "
            f"risk={self.risk_level}, "
            f"obs={self._total_observations:.1f})"
        )
