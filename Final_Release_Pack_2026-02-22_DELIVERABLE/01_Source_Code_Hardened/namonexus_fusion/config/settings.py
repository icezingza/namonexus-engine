"""
Configuration dataclass for NamoNexus Fusion Engine.
Patent-Pending Technology | NamoNexus Research Team
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from ..core.constants import (
    DEFAULT_PRIOR_STRENGTH,
    DEFAULT_MAX_TRIALS,
    DEFAULT_CONFIDENCE_SCALE,
    GOLDEN_RATIO,
)


@dataclass
class FusionConfig:
    """
    Configuration for GoldenBayesianFusion and all derived engines.

    Parameters
    ----------
    prior_strength:
        Scalar *s* that sets α₀ = φ · s and β₀ = s.
        Higher values make the prior stronger (more observations needed to move).
    max_trials:
        Maximum number of pseudo-observations per update.
        Controls how aggressively a high-confidence observation shifts the posterior.
    confidence_scale:
        Multiplier converting raw confidence [0,1] to pseudo-trial count.
        n_trials = confidence * confidence_scale, clipped to max_trials.
    strict_validation:
        If True, raise on out-of-range inputs.  If False, clamp silently.
    max_history:
        Maximum number of update records kept in history.
    """

    prior_strength: float = DEFAULT_PRIOR_STRENGTH
    max_trials: int = DEFAULT_MAX_TRIALS
    min_trials: int = 1
    confidence_scale: float = DEFAULT_CONFIDENCE_SCALE
    use_nonlinear_confidence: bool = False
    nonlinear_k: float = 2.0
    risk_low: float = 0.30
    risk_high: float = 0.70
    strict_validation: bool = True
    max_history: int = 1000

    def __post_init__(self) -> None:
        if self.prior_strength <= 0:
            raise ValueError(f"prior_strength must be > 0, got {self.prior_strength}")
        if self.max_trials <= 0:
            raise ValueError(f"max_trials must be > 0, got {self.max_trials}")
        if self.confidence_scale <= 0:
            raise ValueError(f"confidence_scale must be > 0, got {self.confidence_scale}")

    @property
    def alpha0(self) -> float:
        """α₀ = φ · prior_strength"""
        return GOLDEN_RATIO * self.prior_strength

    @property
    def beta0(self) -> float:
        """β₀ = prior_strength"""
        return self.prior_strength

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FusionConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
