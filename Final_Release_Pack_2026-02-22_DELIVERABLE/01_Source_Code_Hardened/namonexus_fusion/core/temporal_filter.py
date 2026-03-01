"""
Temporal Bayesian Filtering — Feature 1.1
==========================================
Patent-Pending Technology | NamoNexus Research Team

Problem
-------
The base GoldenBayesianFusion engine accumulates evidence indefinitely.
Once α and β are updated, old observations carry the same weight as new
ones forever.  In real-world use this causes two concrete failures:

  1. **State lag** — if a person's emotional state changes (calm → stressed),
     the posterior moves too slowly because the past "drags" it back.
  2. **Session contamination** — early noise corrupts the entire session.

Solution: Exponential Forgetting (Discounted Bayesian Update)
-------------------------------------------------------------
Before each new observation, we apply a decay factor λ ∈ (0, 1] to the
accumulated evidence (not the prior):

    α_eff = α₀  +  λ · (α − α₀)
    β_eff = β₀  +  λ · (β − β₀)

This keeps the prior (α₀, β₀) intact and only discounts the *accumulated
observations*.  The effective number of observations shrinks by λ each
step, so recent observations carry exponentially more weight.

Patent Claim (new — Claim 6)
-----------------------------
"A method of applying a Golden Ratio-anchored forgetting factor to a
Bayesian multimodal fusion posterior, wherein the decay parameter λ is
initialized as the reciprocal of the Golden Ratio (λ₀ = 1/φ ≈ 0.618)
and may be adapted per-modality or per-session, such that the prior
parameters α₀/β₀ = φ are preserved exactly under any decay schedule."

Design Decisions
----------------
* λ = 1.0   → no forgetting (identical to base engine)
* λ = 1/φ   → Golden Ratio decay (default, patent anchor)
* λ = 0.9   → slower forgetting (~10 obs half-life)
* λ = 0.5   → aggressive (~1 obs half-life)

The TemporalFilter is a **mixin / wrapper** — it does not replace
GoldenBayesianFusion but wraps it, keeping the two concerns separate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.constants import GOLDEN_RATIO, GOLDEN_RATIO_RECIPROCAL
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalConfig:
    """
    Configuration for the Temporal Bayesian Filter.

    Parameters
    ----------
    decay_factor:
        λ ∈ (0, 1].  Applied to accumulated evidence before each update.
        Default = 1/φ ≈ 0.618 (Golden Ratio reciprocal — patent anchor).
        Set to 1.0 to disable forgetting entirely.
    adaptive_decay:
        If True, λ is adjusted automatically based on the *rate of change*
        of the fused score.  Fast change → smaller λ (forget faster).
    adaptive_sensitivity:
        Controls how aggressively λ is reduced when change is detected.
        Higher = faster adaptation.
    time_based_decay:
        If True, apply additional decay proportional to elapsed wall-clock
        time between observations (useful for async / real-world pipelines).
    time_decay_rate:
        Decay per second when time_based_decay is enabled.
        λ_effective = λ · exp(−time_decay_rate · Δt)
    window_size:
        Maximum number of recent observations used for adaptive decay
        calculation.  0 = use all history.
    """

    decay_factor: float = GOLDEN_RATIO_RECIPROCAL   # 1/φ ≈ 0.618
    adaptive_decay: bool = False
    adaptive_sensitivity: float = 2.0
    time_based_decay: bool = False
    time_decay_rate: float = 0.01                   # per second
    window_size: int = 20

    def __post_init__(self) -> None:
        if not (0.0 < self.decay_factor <= 1.0):
            raise ConfigurationError(
                f"decay_factor must be in (0, 1], got {self.decay_factor}."
            )
        if self.adaptive_sensitivity <= 0:
            raise ConfigurationError(
                f"adaptive_sensitivity must be > 0, got {self.adaptive_sensitivity}."
            )
        if self.time_decay_rate < 0:
            raise ConfigurationError(
                f"time_decay_rate must be >= 0, got {self.time_decay_rate}."
            )

    @classmethod
    def golden(cls) -> "TemporalConfig":
        """Default: λ = 1/φ, no adaptive or time-based decay."""
        return cls(decay_factor=GOLDEN_RATIO_RECIPROCAL)

    @classmethod
    def adaptive(cls, sensitivity: float = 2.0) -> "TemporalConfig":
        """Adaptive λ that reacts to score velocity."""
        return cls(
            decay_factor=GOLDEN_RATIO_RECIPROCAL,
            adaptive_decay=True,
            adaptive_sensitivity=sensitivity,
        )

    @classmethod
    def realtime(cls, decay_rate: float = 0.01) -> "TemporalConfig":
        """Time-based decay for real-time async pipelines."""
        return cls(
            decay_factor=GOLDEN_RATIO_RECIPROCAL,
            time_based_decay=True,
            time_decay_rate=decay_rate,
        )

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# Temporal state snapshot
# ---------------------------------------------------------------------------


@dataclass
class TemporalSnapshot:
    """One timestep in the temporal history."""

    timestamp: float
    alpha: float
    beta: float
    fused_score: float
    lambda_applied: float
    modality: Optional[str] = None

    @property
    def age(self) -> float:
        """Seconds since this snapshot was captured."""
        return time.time() - self.timestamp


# ---------------------------------------------------------------------------
# Core filter
# ---------------------------------------------------------------------------


class TemporalBayesianFilter:
    """
    Wraps a pair of (α, β) parameters and applies exponential forgetting
    before each update, while preserving the Golden Ratio prior exactly.

    This class is designed to be **composed into** GoldenBayesianFusion
    (see ``TemporalGoldenFusion``) rather than used standalone.

    Parameters
    ----------
    alpha0, beta0:
        Prior parameters (must satisfy alpha0/beta0 = φ).
    config:
        Temporal filter configuration.
    """

    def __init__(
        self,
        alpha0: float,
        beta0: float,
        config: Optional[TemporalConfig] = None,
    ) -> None:
        if alpha0 <= 0 or beta0 <= 0:
            raise ConfigurationError("alpha0 and beta0 must be > 0.")

        self._alpha0: float = alpha0
        self._beta0:  float = beta0
        self._config: TemporalConfig = config or TemporalConfig.golden()

        # Live state
        self._alpha: float = alpha0
        self._beta:  float = beta0
        self._last_update_time: Optional[float] = None
        self._score_history: List[float] = []
        self._snapshots:     List[TemporalSnapshot] = []

        logger.info(
            "TemporalBayesianFilter | α₀=%.4f β₀=%.4f λ=%.4f adaptive=%s",
            alpha0, beta0,
            self._config.decay_factor,
            self._config.adaptive_decay,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def config(self) -> TemporalConfig:
        return self._config

    @property
    def snapshots(self) -> List[TemporalSnapshot]:
        return list(self._snapshots)

    @property
    def effective_lambda(self) -> float:
        """Current effective decay factor (may differ from config if adaptive)."""
        return self._compute_lambda()

    # ------------------------------------------------------------------
    # Internal: decay computation
    # ------------------------------------------------------------------

    def _compute_lambda(
        self,
        current_score: Optional[float] = None,
        now: Optional[float] = None,
    ) -> float:
        """
        Compute the effective λ for the *next* update.

        Combines three sources:
          1. Base λ from config
          2. Adaptive correction based on score velocity (if enabled)
          3. Time-based additional decay (if enabled)
        """
        lam = self._config.decay_factor

        # 2. Adaptive: reduce λ when score is changing fast
        if self._config.adaptive_decay and current_score is not None:
            velocity = self._score_velocity(current_score)
            # Larger velocity → smaller λ (forget faster)
            reduction = np.tanh(self._config.adaptive_sensitivity * velocity)
            lam = lam * (1.0 - 0.5 * reduction)   # max 50% reduction
            lam = max(lam, 0.1)                     # floor at 0.1

        # 3. Time-based: additional decay proportional to elapsed time
        if self._config.time_based_decay and self._last_update_time is not None:
            now = now or time.time()
            dt = now - self._last_update_time
            time_factor = np.exp(-self._config.time_decay_rate * dt)
            lam = lam * time_factor
            lam = max(lam, 0.05)                    # floor at 0.05

        return float(np.clip(lam, 0.05, 1.0))

    def _score_velocity(self, current_score: float) -> float:
        """
        Estimate rate of change of the fused score.

        Uses the last ``window_size`` score values.
        Returns a value in [0, 1] (normalised absolute change rate).
        """
        w = self._config.window_size
        hist = self._score_history[-w:] if w > 0 else self._score_history
        if len(hist) < 2:
            return 0.0
        diffs = [abs(hist[i] - hist[i - 1]) for i in range(1, len(hist))]
        return float(np.clip(np.mean(diffs) * 10, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Core operation: apply decay then update
    # ------------------------------------------------------------------

    def apply_decay(
        self,
        successes: float,
        failures:  float,
        current_score: float,
        modality: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Apply forgetting factor and add new evidence.

        Steps
        -----
        1. Compute effective λ
        2. Decay *accumulated evidence* only (prior preserved):
               α_eff = α₀ + λ · (α − α₀)
               β_eff = β₀ + λ · (β − β₀)
        3. Add new evidence:
               α ← α_eff + successes
               β ← β_eff + failures
        4. Record snapshot

        Parameters
        ----------
        successes, failures:
            Evidence from the current observation (already computed
            by the parent engine's confidence-to-trials mapping).
        current_score:
            Fused score *before* this update (used for velocity calc).
        modality:
            Optional modality label for logging.

        Returns
        -------
        (new_alpha, new_beta)
        """
        now = time.time()
        lam = self._compute_lambda(current_score=current_score, now=now)

        # Step 2: decay accumulated evidence, preserve prior
        alpha_eff = self._alpha0 + lam * (self._alpha - self._alpha0)
        beta_eff  = self._beta0  + lam * (self._beta  - self._beta0)

        # Step 3: add new evidence
        new_alpha = alpha_eff + successes
        new_beta  = beta_eff  + failures

        self._alpha = new_alpha
        self._beta  = new_beta

        # Bookkeeping
        self._score_history.append(current_score)
        if len(self._score_history) > max(self._config.window_size * 2, 100):
            self._score_history.pop(0)

        snap = TemporalSnapshot(
            timestamp=now,
            alpha=new_alpha,
            beta=new_beta,
            fused_score=new_alpha / (new_alpha + new_beta),
            lambda_applied=lam,
            modality=modality,
        )
        self._snapshots.append(snap)
        if len(self._snapshots) > 500:
            self._snapshots.pop(0)

        self._last_update_time = now

        logger.debug(
            "temporal_decay | λ=%.4f α: %.4f→%.4f β: %.4f→%.4f mod=%s",
            lam,
            self._alpha0 + (self._alpha - self._alpha0) / lam if lam > 0 else 0,
            new_alpha, beta_eff, new_beta, modality or "?",
        )

        return new_alpha, new_beta

    def reset(self) -> None:
        """Reset to prior state."""
        self._alpha = self._alpha0
        self._beta  = self._beta0
        self._last_update_time = None
        self._score_history.clear()
        self._snapshots.clear()
        logger.debug("TemporalBayesianFilter reset.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def lambda_history(self) -> List[float]:
        """Return λ values from all recorded snapshots."""
        return [s.lambda_applied for s in self._snapshots]

    def score_trajectory(self) -> List[Tuple[float, float]]:
        """Return list of (timestamp, fused_score) from snapshots."""
        return [(s.timestamp, s.fused_score) for s in self._snapshots]

    def effective_observation_count(self) -> float:
        """
        Estimate how many 'equivalent fresh observations' are currently
        represented in the posterior.

        Computed as (α + β − α₀ − β₀), i.e. accumulated evidence only.
        """
        return max(0.0, (self._alpha + self._beta) - (self._alpha0 + self._beta0))

    def __repr__(self) -> str:
        return (
            f"TemporalBayesianFilter("
            f"α={self._alpha:.4f}, β={self._beta:.4f}, "
            f"λ={self._config.decay_factor:.4f}, "
            f"eff_obs={self.effective_observation_count():.1f})"
        )
