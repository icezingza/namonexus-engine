"""
Drift Detection — Feature 3.1
===============================
Patent-Pending Technology | NamoNexus Research Team

Problem
-------
Sensor distributions shift over time due to:

  • Model updates (e.g., face model retrained, altering score distributions)
  • Hardware aging (microphone degrades, face camera develops noise)
  • Environment changes (different lighting, background noise)
  • Data drift (user population changes, new use cases)

Without drift detection, the fusion engine silently produces degraded
output — posterior intervals drift, risk classifications become
unreliable, and the system loses calibration.  By the time an operator
notices something is wrong, many sessions may already be corrupted.

Solution: Page-Hinkley Test with Golden Ratio-Weighted Control Limits
----------------------------------------------------------------------
The Page-Hinkley (PH) test is a sequential change-point detection
algorithm well-suited to streaming data.  For a series of consistency
values c₁, c₂, ...:

    Cumulative mean:  M_n  = (1/n) Σᵢ cᵢ
    PH statistic:     U_n  = Σᵢ (cᵢ − M_n + δ)     (upward variant)
                      PH_n = U_n − min(U₁..Uₙ)

Drift is signaled when PH_n > threshold (h).

We use the *downward* variant to detect a sustained *drop* in
consistency (i.e., a sensor becoming less reliable over time):

    D_n  = Σᵢ (M_n − cᵢ + δ)
    PHd_n = D_n − min(D₁..Dₙ)

Patent innovation: the threshold h and sensitivity δ are initialized
using the Golden Ratio:

    h = φ² ≈ 2.618       (control limit — generous but bounded)
    δ = 1/φ² ≈ 0.382     (sensitivity — reciprocal of control limit)

These values are then *adapted* based on the posterior uncertainty U
of the fusion engine:

    h_eff = h × (1 + U × φ)

When the engine is uncertain (high U), the control limit *widens*
(we require stronger evidence before alarming), preventing false
positives during periods of genuine state change.

Patent Claim (new — Claim 11)
------------------------------
"A method of detecting statistical drift in a multimodal Bayesian fusion
engine, comprising:
(a) maintaining a per-modality Page-Hinkley statistic on the consistency
    series between individual modality posteriors and the aggregate
    posterior;
(b) initializing the detection threshold h and sensitivity δ as Golden
    Ratio-derived constants h = φ² and δ = 1/φ²;
(c) adapting the effective threshold h_eff = h × (1 + U × φ), where U
    is the current posterior uncertainty of the fusion engine, such that
    the detection threshold widens proportionally to fusion uncertainty;
(d) emitting a DriftEvent when PHd_n > h_eff, annotated with the
    affected modality, detected change magnitude, and uncertainty context;
such that the system distinguishes genuine sensor drift from normal
posterior uncertainty fluctuations."

Design Notes
------------
* Separate PH accumulators per modality: drift in one sensor does not
  mask or trigger alarms for another.
* Soft vs hard alarms: WARNING at h_eff × φ_reciprocal, ALARM at h_eff.
* Auto-reset: after ALARM, accumulator resets and a cooldown begins.
  The sensor is monitored from a fresh baseline after the cooldown.
* Thread-safe: all state mutations are protected by a reentrant lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Golden Ratio constants (local copies for zero-dependency usage)
# ---------------------------------------------------------------------------

_PHI: float = (1.0 + 5.0 ** 0.5) / 2.0   # ≈ 1.618
_PHI_SQ: float = _PHI ** 2                 # ≈ 2.618  → default threshold h
_PHI_SQ_RECIP: float = 1.0 / _PHI_SQ      # ≈ 0.382  → default sensitivity δ
_PHI_RECIP: float = 1.0 / _PHI             # ≈ 0.618


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DriftSeverity(Enum):
    """Severity levels for drift events."""
    NORMAL  = "normal"
    WARNING = "warning"    # PH_n > h_eff × φ_reciprocal
    ALARM   = "alarm"      # PH_n > h_eff  (full threshold exceeded)
    RESET   = "reset"      # Accumulator reset after ALARM + cooldown


class DriftDirection(Enum):
    """Direction of detected drift."""
    UPWARD   = "upward"    # Consistency unexpectedly increased (unusual)
    DOWNWARD = "downward"  # Consistency dropped (sensor degrading — typical)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DriftEvent:
    """
    A structured drift detection event emitted by DriftDetector.

    Fields
    ------
    modality:
        Name of the affected sensor/modality.
    severity:
        NORMAL / WARNING / ALARM / RESET.
    direction:
        UPWARD or DOWNWARD.
    ph_statistic:
        Raw Page-Hinkley statistic at the time of the event.
    effective_threshold:
        h_eff = h × (1 + U × φ) at the time of the event.
    running_mean:
        Rolling mean of the consistency series at this point.
    posterior_uncertainty:
        U from the fusion engine at the time of this event.
    n_observations:
        Number of observations processed by this detector.
    timestamp:
        Wall-clock time of the event (seconds since epoch).
    metadata:
        Optional extra context (session_id, subject_id, etc.).
    """

    modality:             str
    severity:             DriftSeverity
    direction:            DriftDirection
    ph_statistic:         float
    effective_threshold:  float
    running_mean:         float
    posterior_uncertainty: float
    n_observations:       int
    timestamp:            float = field(default_factory=time.time)
    metadata:             Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"]  = self.severity.value
        d["direction"] = self.direction.value
        return d

    def __repr__(self) -> str:
        return (
            f"DriftEvent({self.severity.value.upper()} | mod={self.modality} "
            f"PH={self.ph_statistic:.4f} h_eff={self.effective_threshold:.4f} "
            f"mean={self.running_mean:.4f} U={self.posterior_uncertainty:.4f})"
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftConfig:
    """
    Configuration for a single modality's Page-Hinkley drift detector.

    Parameters
    ----------
    threshold_h:
        Control limit h.  Default = φ² ≈ 2.618 (Golden Ratio squared).
        Alarm fires when PHd_n > h_eff = h × (1 + U × φ).
    sensitivity_delta:
        Allowable slack δ per observation.  Default = 1/φ² ≈ 0.382.
        Larger δ → slower to alarm, tolerates more noise.
    uncertainty_scaling:
        Multiplier on the uncertainty term in h_eff.
        h_eff = h × (1 + uncertainty × uncertainty_scaling × φ).
        Default = 1.0 (standard Golden Ratio scaling).
    min_observations:
        Minimum observations before the alarm can fire.
        Prevents false positives during warm-up.
    cooldown_observations:
        After an ALARM + reset, how many observations to skip before
        re-arming the detector.
    window_size:
        Rolling window for the running mean.  0 = cumulative mean.
    emit_warnings:
        If True, emit WARNING events at h_eff × φ_reciprocal.
    auto_reset:
        If True, automatically reset accumulator after ALARM.
    """

    threshold_h:          float = _PHI_SQ         # φ² ≈ 2.618
    sensitivity_delta:    float = _PHI_SQ_RECIP    # 1/φ² ≈ 0.382
    uncertainty_scaling:  float = 1.0
    min_observations:     int   = 10
    cooldown_observations: int  = 20
    window_size:          int   = 50
    emit_warnings:        bool  = True
    auto_reset:           bool  = True

    def __post_init__(self) -> None:
        if self.threshold_h <= 0:
            raise ValueError(f"threshold_h must be > 0, got {self.threshold_h}.")
        if self.sensitivity_delta < 0:
            raise ValueError(f"sensitivity_delta must be >= 0, got {self.sensitivity_delta}.")
        if self.min_observations < 1:
            raise ValueError(f"min_observations must be >= 1, got {self.min_observations}.")

    @classmethod
    def golden(cls) -> "DriftConfig":
        """Default: φ²-threshold, 1/φ²-sensitivity, standard scaling."""
        return cls()

    @classmethod
    def sensitive(cls) -> "DriftConfig":
        """Tighter control limits — alarms faster, more false positives."""
        return cls(
            threshold_h=_PHI,          # φ ≈ 1.618
            sensitivity_delta=_PHI_SQ_RECIP / 2.0,
            min_observations=5,
        )

    @classmethod
    def conservative(cls) -> "DriftConfig":
        """Wider control limits — fewer false positives, slower detection."""
        return cls(
            threshold_h=_PHI_SQ * _PHI,   # φ³ ≈ 4.236
            sensitivity_delta=_PHI_SQ_RECIP,
            min_observations=20,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-modality PH accumulator
# ---------------------------------------------------------------------------


class _PHAccumulator:
    """
    Stateful Page-Hinkley accumulator for one modality.

    Implements the *downward* variant to detect a sustained DROP in
    consistency below an estimated baseline mean mu0.

    Correct PH formulation
    ----------------------
    The PH test requires a FIXED reference mean mu0 — not a rolling mean.
    Using a rolling mean causes the statistic to grow monotonically for
    stationary inputs (false positives), because the delta term accumulates
    even when consistency == mean_n.

    We estimate mu0 from the first ``min_observations`` values (warm-up),
    then keep it fixed.  After an ALARM + auto-reset, mu0 is re-estimated.

    Downward variant:
        D_n   = SUM_{i=1}^{n} (mu0 - c_i + delta)
        PHd_n = D_n - min(D_1..D_n)
        Alarm when PHd_n > h_eff

    For a stationary sequence (c_i ~ mu0), the expected per-step increment
    is delta, but D_min tracks D_n closely, keeping PHd_n bounded near 0.
    PHd_n grows only when c_i << mu0 systematically.

    Thread-safe via an internal RLock.
    """

    def __init__(self, modality: str, config: DriftConfig) -> None:
        self.modality = modality
        self.config   = config
        self._lock    = threading.RLock()
        self._alarm_count_total: int = 0  # survives reset
        self._reset()

    def _reset(self) -> None:
        """Reset accumulator to initial (pre-arm) state."""
        self._n:       int   = 0
        self._warmup:  list  = []              # observations during warm-up
        self._mu0:     Optional[float] = None  # fixed baseline mean
        self._D:       float = 0.0             # CUSUM statistic
        self._armed:   bool  = False           # True once baseline established
        self._cooldown: int  = 0              # observations remaining in cooldown
        logger.debug("PHAccumulator reset | mod=%s", self.modality)

    def update(self, consistency: float, uncertainty: float) -> Optional[DriftEvent]:
        """
        Process one consistency observation.

        Parameters
        ----------
        consistency:
            Value in [0, 1].  1 = perfectly consistent with aggregate.
        uncertainty:
            Current posterior uncertainty from the fusion engine.

        Returns
        -------
        DriftEvent or None.
        """
        with self._lock:
            # --- Cooldown: skip after reset ---
            if self._cooldown > 0:
                self._cooldown -= 1
                return None

            self._n += 1

            # --- Warm-up phase: build baseline ---
            if not self._armed:
                self._warmup.append(consistency)
                if len(self._warmup) >= self.config.min_observations:
                    self._mu0   = float(np.mean(self._warmup))
                    self._armed = True
                    self._warmup.clear()
                    self._D     = 0.0
                    logger.debug(
                        "PHAccumulator armed | mod=%s mu0=%.4f", self.modality, self._mu0
                    )
                return None  # Never alarm during warm-up

            # --- Armed: run CUSUM (one-sided) against fixed mu0 ---
            assert self._mu0 is not None
            delta = self.config.sensitivity_delta

            # Downward CUSUM: accumulates when c_i < mu0 - delta
            # max(0, ...) clipping prevents indefinite growth on stationary input
            self._D = max(0.0, self._D + (self._mu0 - consistency) - delta)
            ph_stat = self._D

            # Effective threshold: h * (1 + U * phi * scale)
            h_eff = self.config.threshold_h * (
                1.0 + uncertainty * _PHI * self.config.uncertainty_scaling
            )

            # --- WARNING ---
            warning_threshold = h_eff * _PHI_RECIP
            if self.config.emit_warnings and warning_threshold < ph_stat <= h_eff:
                return DriftEvent(
                    modality=self.modality,
                    severity=DriftSeverity.WARNING,
                    direction=DriftDirection.DOWNWARD,
                    ph_statistic=ph_stat,
                    effective_threshold=h_eff,
                    running_mean=self._mu0,
                    posterior_uncertainty=uncertainty,
                    n_observations=self._n,
                )

            # --- ALARM ---
            if ph_stat > h_eff:
                self._alarm_count_total += 1
                count = self._alarm_count_total
                alarm_event = DriftEvent(
                    modality=self.modality,
                    severity=DriftSeverity.ALARM,
                    direction=DriftDirection.DOWNWARD,
                    ph_statistic=ph_stat,
                    effective_threshold=h_eff,
                    running_mean=self._mu0,
                    posterior_uncertainty=uncertainty,
                    n_observations=self._n,
                    metadata={"alarm_count": count},
                )

                if self.config.auto_reset:
                    self._reset()
                    self._alarm_count_total = count  # preserve across reset
                    self._cooldown = self.config.cooldown_observations
                    logger.info(
                        "DriftDetector ALARM + reset | mod=%s alarm_count=%d",
                        self.modality, count,
                    )

                return alarm_event

            return None

    @property
    def n_observations(self) -> int:
        with self._lock:
            return self._n

    @property
    def alarm_count(self) -> int:
        with self._lock:
            return self._alarm_count_total

    @property
    def is_in_cooldown(self) -> bool:
        with self._lock:
            return self._cooldown > 0


# ---------------------------------------------------------------------------
# Multi-modality drift detector
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Manages per-modality Page-Hinkley drift detection for the fusion engine.

    Architecture
    ------------
    One ``_PHAccumulator`` is created lazily per modality as observations
    arrive.  Each accumulator maintains its own rolling statistics and
    threshold, independent of the others.

    Thread-safety
    -------------
    Safe for concurrent ``update()`` calls.  Accumulators are created
    under a global lock; individual accumulator updates use their own
    per-modality locks.

    Parameters
    ----------
    config:
        Default DriftConfig applied to all modalities.
    per_modality_config:
        Optional per-modality overrides.  Keys are modality names.
    callbacks:
        List of callables invoked when a DriftEvent is emitted:
        ``callback(event: DriftEvent) → None``.

    Examples
    --------
    ::

        detector = DriftDetector()

        # In the fusion update loop:
        event = detector.update("voice", consistency=0.45, uncertainty=0.12)
        if event and event.severity == DriftSeverity.ALARM:
            print(f"Drift detected in {event.modality}!")

        # Access diagnostics:
        print(detector.summary())
    """

    def __init__(
        self,
        config:               Optional[DriftConfig]             = None,
        per_modality_config:  Optional[Dict[str, DriftConfig]]  = None,
        callbacks:            Optional[List[Callable[[DriftEvent], None]]] = None,
    ) -> None:
        self._default_config    = config or DriftConfig.golden()
        self._per_mod_config    = per_modality_config or {}
        self._callbacks         = callbacks or []
        self._accumulators:     Dict[str, _PHAccumulator] = {}
        self._events:           List[DriftEvent] = []
        self._global_lock       = threading.Lock()

        logger.info(
            "DriftDetector | h=%.4f δ=%.4f min_obs=%d window=%d",
            self._default_config.threshold_h,
            self._default_config.sensitivity_delta,
            self._default_config.min_observations,
            self._default_config.window_size,
        )

    # ------------------------------------------------------------------
    # Modality management
    # ------------------------------------------------------------------

    def _get_accumulator(self, modality: str) -> _PHAccumulator:
        """Retrieve or lazily create an accumulator for *modality*."""
        with self._global_lock:
            if modality not in self._accumulators:
                cfg = self._per_mod_config.get(modality, self._default_config)
                self._accumulators[modality] = _PHAccumulator(modality, cfg)
                logger.debug("Created PHAccumulator for modality=%s", modality)
            return self._accumulators[modality]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def update(
        self,
        modality:     str,
        consistency:  float,
        uncertainty:  float,
        metadata:     Optional[Dict[str, Any]] = None,
    ) -> Optional[DriftEvent]:
        """
        Process one consistency observation for *modality*.

        Parameters
        ----------
        modality:
            Sensor/modality name (e.g., "voice", "face", "text").
        consistency:
            How consistent this observation was with the aggregate posterior.
            Computed externally as: 1 − |score_m − fused_score_before|.
        uncertainty:
            Current posterior uncertainty (σ or variance proxy) from the
            fusion engine.  Used to adapt the effective threshold h_eff.
        metadata:
            Optional extra context attached to any emitted DriftEvent.

        Returns
        -------
        DriftEvent or None.
            If a WARNING or ALARM is triggered, returns the event.
            Otherwise returns None.
        """
        consistency = float(np.clip(consistency, 0.0, 1.0))
        uncertainty = float(max(0.0, uncertainty))

        acc   = self._get_accumulator(modality)
        event = acc.update(consistency, uncertainty)

        if event is not None:
            if metadata:
                event.metadata.update(metadata)
            with self._global_lock:
                self._events.append(event)

            for cb in self._callbacks:
                try:
                    cb(event)
                except Exception as exc:
                    logger.warning("DriftDetector callback error: %s", exc)

            logger.info("DriftEvent: %s", event)

        return event

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def events(
        self,
        modality:  Optional[str]          = None,
        severity:  Optional[DriftSeverity] = None,
        since:     Optional[float]         = None,
    ) -> List[DriftEvent]:
        """
        Return filtered drift events.

        Parameters
        ----------
        modality:
            Filter by modality name.
        severity:
            Filter by severity.
        since:
            Unix timestamp; only return events after this time.
        """
        with self._global_lock:
            evts = list(self._events)

        if modality is not None:
            evts = [e for e in evts if e.modality == modality]
        if severity is not None:
            evts = [e for e in evts if e.severity == severity]
        if since is not None:
            evts = [e for e in evts if e.timestamp >= since]
        return evts

    def alarm_count(self, modality: Optional[str] = None) -> int:
        """Total ALARMs across all (or one) modality."""
        with self._global_lock:
            accs = (
                [self._accumulators[modality]]
                if modality and modality in self._accumulators
                else list(self._accumulators.values())
            )
        return sum(a.alarm_count for a in accs)

    def is_in_cooldown(self, modality: str) -> bool:
        """True if *modality* is currently in post-alarm cooldown."""
        with self._global_lock:
            if modality not in self._accumulators:
                return False
            return self._accumulators[modality].is_in_cooldown

    def modalities_monitored(self) -> List[str]:
        """List of all modalities with active accumulators."""
        with self._global_lock:
            return list(self._accumulators.keys())

    def reset(self, modality: Optional[str] = None) -> None:
        """
        Reset accumulators.

        Parameters
        ----------
        modality:
            If given, reset only that modality; otherwise reset all.
        """
        with self._global_lock:
            if modality:
                if modality in self._accumulators:
                    self._accumulators[modality]._reset()
            else:
                for acc in self._accumulators.values():
                    acc._reset()
                self._events.clear()

    def summary(self) -> Dict[str, Any]:
        """Return a diagnostic summary across all monitored modalities."""
        with self._global_lock:
            mods = list(self._accumulators.items())

        rows = []
        for name, acc in mods:
            rows.append({
                "modality":         name,
                "n_observations":   acc.n_observations,
                "alarm_count":      acc.alarm_count,
                "in_cooldown":      acc.is_in_cooldown,
            })

        total_alarms = sum(r["alarm_count"] for r in rows)
        return {
            "modalities":   rows,
            "total_alarms": total_alarms,
            "total_events": len(self._events),
            "config":       self._default_config.to_dict(),
        }

    def add_callback(self, callback: Callable[[DriftEvent], None]) -> None:
        """Register a new event callback."""
        self._callbacks.append(callback)

    def __repr__(self) -> str:
        n_mods = len(self._accumulators)
        n_alarms = self.alarm_count()
        return (
            f"DriftDetector(modalities={n_mods}, alarms={n_alarms}, "
            f"h={self._default_config.threshold_h:.3f}, "
            f"δ={self._default_config.sensitivity_delta:.3f})"
        )
