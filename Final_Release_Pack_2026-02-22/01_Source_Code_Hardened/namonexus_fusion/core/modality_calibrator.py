"""
Modality Reliability Auto-Calibration — Feature 2.1
=====================================================
Patent-Pending Technology | NamoNexus Research Team

Problem
-------
Every modality is treated with equal trust — only the *caller-provided*
confidence score differentiates them.  In practice:

  • A microphone in a noisy environment is systematically less reliable
    than the same microphone in a quiet room.
  • A face-detection model trained on adults performs poorly on children.
  • Any modality can drift over time (hardware aging, distribution shift).

The engine has no mechanism to detect or compensate for these failures
automatically.  A caller providing confidence=0.9 for a degraded sensor
will corrupt the posterior.

Solution: Bayesian Consistency Calibration
------------------------------------------
After each observation from modality *m*, we measure how *consistent*
that observation was with the current multimodal posterior.  Consistency
is defined as:

    consistency_m = 1 − |score_m − fused_score_before|

A modality that repeatedly contradicts the aggregate is penalised; one
that aligns consistently is rewarded.

We maintain a Beta distribution Beta(τ_α, τ_β) per modality — the
**trust posterior** — and update it after each observation:

    if consistency ≥ threshold:
        τ_α ← τ_α + consistency        (reward)
    else:
        τ_β ← τ_β + (1 − consistency)  (penalise)

The **effective confidence** fed into the fusion engine is then:

    effective_confidence = raw_confidence × trust_mean
    trust_mean = τ_α / (τ_α + τ_β)

Patent Claim (new — Claim 8)
-----------------------------
"A system and method for automatically calibrating the reliability weight
of a sensing modality in a multimodal Bayesian fusion engine, wherein:
(a) a trust posterior Beta(τ_α, τ_β) is maintained per modality,
    initialised with τ_α/τ_β = φ (Golden Ratio);
(b) after each observation, consistency with the aggregate posterior is
    measured and used to update the trust posterior via conjugate update;
(c) the effective confidence supplied to the fusion engine is the product
    of the raw confidence and the trust posterior mean;
such that modalities that are consistently aligned with the aggregate
receive progressively higher weight, and degraded sensors are down-weighted
automatically without manual intervention."

Design Notes
------------
* Trust prior: τ_α₀ = φ, τ_β₀ = 1  → trust_mean ≈ 0.618 initially
  (mild scepticism, same Golden Ratio anchor as main engine)
* Consistency threshold θ: observations above θ are "aligned",
  below θ are "misaligned".  Default θ = 0.3 (configurable).
* Trust is bounded: trust_mean ∈ [trust_floor, 1.0] to prevent
  complete silencing of a modality.
* Temporal decay is applied to trust posteriors as well (optional),
  allowing recovery from a period of poor sensor quality.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.constants import GOLDEN_RATIO, GOLDEN_RATIO_RECIPROCAL
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationConfig:
    """
    Configuration for ModalityCalibrator.

    Parameters
    ----------
    consistency_threshold:
        Minimum consistency score to count as "aligned".
        Default 0.3 — observations within 0.3 of fused score are aligned.
    trust_floor:
        Minimum trust mean allowed.  Prevents a modality from being
        silenced entirely.  Default 0.10.
    trust_decay:
        Forgetting factor applied to trust posteriors each update.
        1.0 = no forgetting (trust accumulates indefinitely).
        < 1 = allows recovery from periods of poor sensor quality.
        Default = 1/φ (Golden Ratio reciprocal).
    initial_trust_strength:
        Controls how strongly the trust prior resists early updates.
        Larger = more conservative initial trust.
    reward_scale:
        Multiplier on the consistency reward.  Default 1.0.
    penalty_scale:
        Multiplier on the inconsistency penalty.  Default 1.0.
        Setting penalty_scale > reward_scale makes the system more
        aggressive at downweighting bad sensors.
    """

    consistency_threshold: float = 0.30
    trust_floor:           float = 0.10
    trust_decay:           float = GOLDEN_RATIO_RECIPROCAL
    initial_trust_strength: float = 1.0
    reward_scale:          float = 1.0
    penalty_scale:         float = 1.0

    def __post_init__(self) -> None:
        errs: List[str] = []
        if not (0.0 < self.consistency_threshold < 1.0):
            errs.append(f"consistency_threshold must be in (0,1), got {self.consistency_threshold}.")
        if not (0.0 <= self.trust_floor < 1.0):
            errs.append(f"trust_floor must be in [0,1), got {self.trust_floor}.")
        if not (0.0 < self.trust_decay <= 1.0):
            errs.append(f"trust_decay must be in (0,1], got {self.trust_decay}.")
        if self.initial_trust_strength <= 0:
            errs.append(f"initial_trust_strength must be > 0, got {self.initial_trust_strength}.")
        if errs:
            raise ConfigurationError("Invalid CalibrationConfig:\n" + "\n".join(f"  • {e}" for e in errs))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-modality trust state
# ---------------------------------------------------------------------------


@dataclass
class ModalityTrustState:
    """
    Tracks the trust history and current Beta posterior for one modality.
    """

    modality:    str
    tau_alpha:   float          # trust posterior α
    tau_beta:    float          # trust posterior β
    n_updates:   int = 0
    n_rewards:   int = 0
    n_penalties: int = 0
    last_consistency: float = 0.0
    created_at:  float = field(default_factory=time.time)

    # Rolling history of (timestamp, trust_mean, consistency) for diagnostics
    history: List[Tuple[float, float, float]] = field(default_factory=list)

    @property
    def trust_mean(self) -> float:
        """Current trust mean = τ_α / (τ_α + τ_β)."""
        return self.tau_alpha / (self.tau_alpha + self.tau_beta)

    @property
    def trust_uncertainty(self) -> float:
        """Uncertainty of the trust estimate (Beta std-dev)."""
        s = self.tau_alpha + self.tau_beta
        return float(np.sqrt(self.tau_alpha * self.tau_beta / (s * s * (s + 1))))

    @property
    def reward_rate(self) -> float:
        """Fraction of updates that were rewards."""
        if self.n_updates == 0:
            return 0.5
        return self.n_rewards / self.n_updates

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality":         self.modality,
            "tau_alpha":        round(self.tau_alpha, 6),
            "tau_beta":         round(self.tau_beta, 6),
            "trust_mean":       round(self.trust_mean, 6),
            "trust_uncertainty":round(self.trust_uncertainty, 6),
            "n_updates":        self.n_updates,
            "reward_rate":      round(self.reward_rate, 4),
        }


# ---------------------------------------------------------------------------
# Core calibrator
# ---------------------------------------------------------------------------


class ModalityCalibrator:
    """
    Automatically calibrates the reliability weight of each sensing
    modality using a per-modality Beta trust posterior initialised with
    the Golden Ratio.

    This class is designed to be **composed into** Phase2GoldenFusion.
    It does not perform fusion itself — it modulates the confidence
    values that are fed into the fusion engine.

    Parameters
    ----------
    config:
        Calibration configuration.

    Examples
    --------
    ::

        calibrator = ModalityCalibrator()

        # Before fusion update:
        eff_conf = calibrator.effective_confidence("voice", raw_conf=0.80,
                                                    fused_score_before=0.62,
                                                    observation_score=0.25)
        # Use eff_conf instead of raw_conf in GoldenBayesianFusion.update()
    """

    def __init__(self, config: Optional[CalibrationConfig] = None) -> None:
        self._config = config or CalibrationConfig()
        self._modalities: Dict[str, ModalityTrustState] = {}

        logger.info(
            "ModalityCalibrator | θ=%.2f floor=%.2f decay=%.4f",
            self._config.consistency_threshold,
            self._config.trust_floor,
            self._config.trust_decay,
        )

    # ------------------------------------------------------------------
    # Internal: initialise a new modality
    # ------------------------------------------------------------------

    def _init_modality(self, modality: str) -> ModalityTrustState:
        """
        Create a new trust state for an unseen modality.

        Trust prior: τ_α₀ = φ · strength, τ_β₀ = strength
        → trust_mean₀ = φ / (φ + 1) = 1/φ ≈ 0.618
        """
        s = self._config.initial_trust_strength
        state = ModalityTrustState(
            modality=modality,
            tau_alpha=GOLDEN_RATIO * s,
            tau_beta=s,
        )
        self._modalities[modality] = state
        logger.debug(
            "New modality registered: %s | initial trust=%.4f",
            modality, state.trust_mean,
        )
        return state

    def _get_or_init(self, modality: str) -> ModalityTrustState:
        return self._modalities.get(modality) or self._init_modality(modality)

    # ------------------------------------------------------------------
    # Consistency measurement
    # ------------------------------------------------------------------

    def _consistency(
        self,
        observation_score: float,
        fused_score_before: float,
    ) -> float:
        """
        Measure how consistent an observation is with the current posterior.

            consistency = 1 − |observation_score − fused_score_before|

        Returns a value in [0, 1].  1 = perfectly aligned, 0 = opposite.
        """
        return float(np.clip(1.0 - abs(observation_score - fused_score_before), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Core public interface
    # ------------------------------------------------------------------

    def effective_confidence(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> float:
        """
        Compute the effective confidence for this observation and update
        the trust posterior.

        This is the **only method** the fusion engine needs to call.

        Parameters
        ----------
        modality:
            Modality label (e.g. ``"voice"``).
        raw_confidence:
            Confidence reported by the sensor/model (before calibration).
        observation_score:
            The score from this observation (used to compute consistency).
        fused_score_before:
            The fused score *before* this update (the current posterior mean).

        Returns
        -------
        float
            Effective confidence in [trust_floor, 1.0], adjusted by
            the current trust mean.
        """
        # Backward compatibility: support both current and legacy call styles.
        # Current:
        #   effective_confidence(modality="text", raw_confidence=0.8,
        #                        observation_score=0.7, fused_score_before=0.6)
        # Legacy keyword:
        #   effective_confidence(raw_confidence=0.8, score=0.7,
        #                        fused_score=0.6, modality="text")
        # Legacy positional:
        #   effective_confidence(0.8, 0.7, 0.6, "text")
        modality: Optional[str] = None
        raw_confidence: Optional[float] = None
        observation_score: Optional[float] = None
        fused_score_before: Optional[float] = None

        if args:
            if len(args) != 4:
                raise TypeError(
                    "effective_confidence expects 4 positional args in legacy mode."
                )
            if isinstance(args[0], str):
                modality = args[0]
                raw_confidence = float(args[1])
                observation_score = float(args[2])
                fused_score_before = float(args[3])
            else:
                raw_confidence = float(args[0])
                observation_score = float(args[1])
                fused_score_before = float(args[2])
                modality = str(args[3])

        modality = kwargs.pop("modality", kwargs.pop("modality_name", modality))
        raw_confidence = kwargs.pop("raw_confidence", raw_confidence)
        if raw_confidence is None:
            raw_confidence = kwargs.pop("confidence", None)
        observation_score = kwargs.pop("observation_score", kwargs.pop("score", observation_score))
        fused_score_before = kwargs.pop("fused_score_before", kwargs.pop("fused_score", fused_score_before))
        if kwargs:
            unknown = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unknown}")
        if modality is None or raw_confidence is None or observation_score is None or fused_score_before is None:
            raise TypeError(
                "effective_confidence requires modality, raw_confidence, observation_score, fused_score_before"
            )
        raw_confidence = float(raw_confidence)
        observation_score = float(observation_score)
        fused_score_before = float(fused_score_before)

        state = self._get_or_init(modality)
        consistency = self._consistency(observation_score, fused_score_before)

        # ── Apply decay to trust posterior ──────────────────────────
        lam = self._config.trust_decay
        tau_alpha0 = GOLDEN_RATIO * self._config.initial_trust_strength
        tau_beta0  = self._config.initial_trust_strength
        state.tau_alpha = tau_alpha0 + lam * (state.tau_alpha - tau_alpha0)
        state.tau_beta  = tau_beta0  + lam * (state.tau_beta  - tau_beta0)

        # ── Update trust posterior ───────────────────────────────────
        if consistency >= self._config.consistency_threshold:
            state.tau_alpha += consistency * self._config.reward_scale
            state.n_rewards += 1
        else:
            state.tau_beta  += (1.0 - consistency) * self._config.penalty_scale
            state.n_penalties += 1

        state.n_updates += 1
        state.last_consistency = consistency

        # Record history (keep last 200 entries)
        state.history.append((time.time(), state.trust_mean, consistency))
        if len(state.history) > 200:
            state.history.pop(0)

        # ── Compute effective confidence ─────────────────────────────
        trust = max(state.trust_mean, self._config.trust_floor)
        eff_conf = float(np.clip(raw_confidence * trust, 0.0, 1.0))

        logger.debug(
            "calibrate | mod=%s raw_conf=%.4f consistency=%.4f trust=%.4f eff_conf=%.4f",
            modality, raw_confidence, consistency, trust, eff_conf,
        )
        return eff_conf

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def modalities(self) -> List[str]:
        """List of all modalities seen so far."""
        return list(self._modalities.keys())

    def trust_state(self, modality: str) -> Optional[ModalityTrustState]:
        """Return the trust state for a modality (None if unseen)."""
        return self._modalities.get(modality)

    def report(self, modality: str) -> Dict[str, Any]:
        """Legacy alias returning one modality trust snapshot."""
        return self._get_or_init(modality).to_dict()

    def trust_summary(self) -> Dict[str, Dict[str, Any]]:
        """Return a summary dict of all modality trust states."""
        return {m: s.to_dict() for m, s in self._modalities.items()}

    def lowest_trust_modality(self) -> Optional[str]:
        """Return the modality with the lowest current trust mean."""
        if not self._modalities:
            return None
        return min(self._modalities, key=lambda m: self._modalities[m].trust_mean)

    def highest_trust_modality(self) -> Optional[str]:
        """Return the modality with the highest current trust mean."""
        if not self._modalities:
            return None
        return max(self._modalities, key=lambda m: self._modalities[m].trust_mean)

    def reset(self, modality: Optional[str] = None) -> None:
        """
        Reset trust state(s).

        Parameters
        ----------
        modality:
            If provided, reset only this modality.
            If None, reset all modalities.
        """
        if modality is not None:
            if modality in self._modalities:
                self._modalities.pop(modality)
        else:
            self._modalities.clear()
        logger.debug("ModalityCalibrator reset: modality=%s", modality or "ALL")

    def __repr__(self) -> str:
        summary = ", ".join(
            f"{m}={s.trust_mean:.3f}"
            for m, s in self._modalities.items()
        )
        return f"ModalityCalibrator([{summary}])"
