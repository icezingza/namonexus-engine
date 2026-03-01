"""
Sensor Trust Scoring — Feature 2.2
====================================
Patent-Pending Technology | NamoNexus Research Team

Relationship to Feature 2.1
----------------------------
ModalityCalibrator (2.1) adjusts *confidence* in real time based on
recent consistency.  SensorTrustScorer (2.2) maintains a **persistent,
long-term trust record** per sensor and can:

  • Detect systematic anomalies (sensor lying consistently)
  • Issue warnings before trust degrades completely
  • Temporarily blacklist a sensor that fails repeatedly
  • Emit structured trust events for external monitoring

The key distinction:

    2.1 (Calibrator) = short-term, per-observation confidence adjustment
    2.2 (Scorer)     = long-term, session-spanning anomaly detection

Patent Claim (new — Claim 9)
-----------------------------
"A method of computing a long-term trust score for each sensing modality
in a multimodal fusion system, wherein:
(a) a rolling consistency record is maintained per sensor over a
    configurable observation window;
(b) anomaly detection is performed using a Page-Hinkley test on the
    consistency series, initialised with a Golden Ratio-derived control
    limit;
(c) a sensor whose cumulative anomaly statistic exceeds a threshold is
    flagged and optionally blacklisted for a configurable cooldown period;
(d) blacklisted sensors contribute zero evidence to the fusion posterior
    until reinstated;
such that the system self-heals by reinstate sensors after the cooldown
period and re-evaluating their consistency from a neutral prior."

Page-Hinkley Test (Brief)
--------------------------
For a series of consistency values c₁, c₂, ...:

    M_n = (1/n) Σ cᵢ           (running mean)
    PH_n = Σ (M_n − cᵢ + δ)   (cumulative sum, upward variant detects drop)

Alarm when PH_n − min(PH_1..PH_n) > threshold.

We use the *downward* variant to detect sustained drops in consistency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..core.constants import GOLDEN_RATIO, GOLDEN_RATIO_RECIPROCAL
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and events
# ---------------------------------------------------------------------------


class TrustLevel(Enum):
    """Qualitative trust classification."""
    HIGH     = "high"
    MODERATE = "moderate"
    LOW      = "low"
    CRITICAL = "critical"
    BLACKLISTED = "blacklisted"


@dataclass
class TrustEvent:
    """
    Structured event emitted when a sensor's trust status changes.

    Suitable for forwarding to monitoring systems (Prometheus, Slack, PagerDuty).
    """

    event_type:   str           # "degraded", "anomaly", "blacklisted", "reinstated"
    modality:     str
    trust_score:  float
    trust_level:  str
    timestamp:    float = field(default_factory=time.time)
    message:      str = ""
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrustScorerConfig:
    """
    Configuration for SensorTrustScorer.

    Parameters
    ----------
    window_size:
        Number of recent observations used for anomaly detection.
    ph_delta:
        Page-Hinkley sensitivity parameter δ.  Smaller = more sensitive.
        Default = 1/φ ≈ 0.618 (Golden Ratio anchor).
    ph_threshold:
        Page-Hinkley alarm threshold.  Larger = fewer false alarms.
        Default = φ ≈ 1.618 (Golden Ratio anchor).
    anomaly_alarm_count:
        Number of consecutive PH alarms before blacklisting.
    blacklist_cooldown_seconds:
        How long a blacklisted sensor is excluded before reinstatement.
    trust_high_threshold:
        Trust score above this → HIGH trust.
    trust_low_threshold:
        Trust score below this → LOW or CRITICAL trust.
    trust_critical_threshold:
        Trust score below this → CRITICAL trust (imminent blacklist).
    event_callbacks:
        Optional list of callables invoked with each TrustEvent.
    """

    window_size:               int   = 30
    ph_delta:                  float = GOLDEN_RATIO_RECIPROCAL  # 1/φ
    ph_threshold:              float = GOLDEN_RATIO             # φ
    anomaly_alarm_count:       int   = 3
    blacklist_cooldown_seconds: float = 60.0
    trust_high_threshold:      float = 0.75
    trust_low_threshold:       float = 0.40
    trust_critical_threshold:  float = 0.20

    def __post_init__(self) -> None:
        errs: List[str] = []
        if self.window_size < 5:
            errs.append(f"window_size must be >= 5, got {self.window_size}.")
        if self.ph_delta <= 0:
            errs.append(f"ph_delta must be > 0, got {self.ph_delta}.")
        if self.ph_threshold <= 0:
            errs.append(f"ph_threshold must be > 0, got {self.ph_threshold}.")
        if errs:
            raise ConfigurationError("Invalid TrustScorerConfig:\n" + "\n".join(f"  • {e}" for e in errs))

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if k != "event_callbacks"}


# ---------------------------------------------------------------------------
# Per-sensor record
# ---------------------------------------------------------------------------


@dataclass
class SensorRecord:
    """Long-term record for one sensor/modality."""

    modality:          str
    trust_score:       float = GOLDEN_RATIO_RECIPROCAL   # start at 1/φ
    n_observations:    int   = 0
    n_anomalies:       int   = 0
    consecutive_alarms: int  = 0
    is_blacklisted:    bool  = False
    blacklisted_at:    Optional[float] = None
    consistency_window: List[float] = field(default_factory=list)
    ph_cumsum:         float = 0.0
    ph_min:            float = 0.0
    events:            List[TrustEvent] = field(default_factory=list)

    @property
    def trust_level(self) -> TrustLevel:
        if self.is_blacklisted:
            return TrustLevel.BLACKLISTED
        if self.trust_score >= 0.75:
            return TrustLevel.HIGH
        if self.trust_score >= 0.40:
            return TrustLevel.MODERATE
        if self.trust_score >= 0.20:
            return TrustLevel.LOW
        return TrustLevel.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality":           self.modality,
            "trust_score":        round(self.trust_score, 6),
            "trust_level":        self.trust_level.value,
            "n_observations":     self.n_observations,
            "n_anomalies":        self.n_anomalies,
            "consecutive_alarms": self.consecutive_alarms,
            "is_blacklisted":     self.is_blacklisted,
        }


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------


class SensorTrustScorer:
    """
    Long-term sensor anomaly detection and trust scoring.

    Complements :class:`ModalityCalibrator` (which handles short-term
    per-observation confidence adjustment) by maintaining persistent
    trust records and detecting systematic sensor failures.

    Parameters
    ----------
    config:
        Scorer configuration.
    event_callbacks:
        Optional list of callables ``(TrustEvent) → None`` invoked
        whenever a trust event is emitted.  Use for monitoring integration.

    Examples
    --------
    ::

        scorer = SensorTrustScorer()

        # After each calibrated observation:
        is_ok = scorer.record_observation("voice", consistency=0.85)
        if not is_ok:
            print("Voice sensor blacklisted — skipping observation")

        print(scorer.trust_report())
    """

    def __init__(
        self,
        config: Optional[TrustScorerConfig] = None,
        event_callbacks: Optional[List[Callable[[TrustEvent], None]]] = None,
    ) -> None:
        self._config    = config or TrustScorerConfig()
        self._callbacks = event_callbacks or []
        self._sensors:  Dict[str, SensorRecord] = {}

        logger.info(
            "SensorTrustScorer | window=%d δ=%.4f threshold=%.4f cooldown=%.0fs",
            self._config.window_size,
            self._config.ph_delta,
            self._config.ph_threshold,
            self._config.blacklist_cooldown_seconds,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_init(self, modality: str) -> SensorRecord:
        if modality not in self._sensors:
            self._sensors[modality] = SensorRecord(modality=modality)
            logger.debug("New sensor registered: %s", modality)
        return self._sensors[modality]

    def _emit_event(self, event: TrustEvent) -> None:
        """Store and broadcast a trust event."""
        rec = self._sensors.get(event.modality)
        if rec:
            rec.events.append(event)
            if len(rec.events) > 100:
                rec.events.pop(0)
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.warning("Event callback error: %s", exc)

    def _update_trust_score(self, rec: SensorRecord) -> None:
        """
        Recompute the trust score from the current consistency window.

            trust_score = mean(window) weighted toward recent values
                        = exponentially weighted mean with α = 1/φ
        """
        if not rec.consistency_window:
            return
        w = len(rec.consistency_window)
        # Exponential weights: most recent gets highest weight
        alpha = GOLDEN_RATIO_RECIPROCAL
        weights = np.array([alpha ** (w - i - 1) for i in range(w)])
        weights /= weights.sum()
        rec.trust_score = float(np.dot(weights, rec.consistency_window))

    def _run_page_hinkley(self, rec: SensorRecord, consistency: float) -> bool:
        """
        Run one step of the CUSUM change-point test.

        Detects consistency that is *persistently low* relative to a
        neutral baseline of 0.5 (mid-range).  This catches both:
          - Sudden drops (sensor just degraded)
          - Chronic low consistency (sensor was always bad)

        CUSUM formula:
            S_n = max(0, S_{n-1} + (baseline − consistency − δ))
        Alarm when S_n > threshold.

        Using a fixed baseline (0.5) rather than a rolling mean ensures
        that a sensor stuck at a bad level is still detected.
        """
        if len(rec.consistency_window) < 5:
            return False

        BASELINE = 0.5   # neutral expected consistency
        increment = (BASELINE - consistency) - self._config.ph_delta
        rec.ph_cumsum = max(0.0, rec.ph_cumsum + increment)

        alarm = rec.ph_cumsum > self._config.ph_threshold

        if alarm:
            rec.ph_cumsum = 0.0
            rec.ph_min    = 0.0

        return alarm

    # ------------------------------------------------------------------
    # Core public interface
    # ------------------------------------------------------------------

    def record_observation(
        self,
        modality: str,
        consistency: float,
    ) -> bool:
        """
        Record a consistency measurement for a sensor and update its
        trust score and anomaly status.

        Parameters
        ----------
        modality:
            Sensor/modality label.
        consistency:
            Consistency of the latest observation with the aggregate
            posterior (output of ``ModalityCalibrator._consistency()``
            or computed externally).  Must be in [0, 1].

        Returns
        -------
        bool
            True  → sensor is active and observation should be used.
            False → sensor is blacklisted and observation must be dropped.
        """
        rec = self._get_or_init(modality)

        # ── Auto-reinstate if cooldown has elapsed ───────────────────
        if rec.is_blacklisted and rec.blacklisted_at is not None:
            elapsed = time.time() - rec.blacklisted_at
            cooldown = self._config.blacklist_cooldown_seconds
            if cooldown > 0 and elapsed >= cooldown:
                self._reinstate(rec)

        if rec.is_blacklisted:
            return False

        # ── Update consistency window ────────────────────────────────
        rec.consistency_window.append(float(np.clip(consistency, 0.0, 1.0)))
        if len(rec.consistency_window) > self._config.window_size:
            rec.consistency_window.pop(0)

        rec.n_observations += 1

        # ── Update trust score ───────────────────────────────────────
        prev_level = rec.trust_level
        self._update_trust_score(rec)
        if rec.trust_score <= self._config.trust_critical_threshold and \
           rec.n_observations >= self._config.anomaly_alarm_count:
            rec.n_anomalies = max(rec.n_anomalies, self._config.anomaly_alarm_count)
            self._blacklist(rec)
            return False

        # ── Page-Hinkley anomaly detection ───────────────────────────
        alarm = self._run_page_hinkley(rec, consistency)

        if alarm:
            rec.n_anomalies        += 1
            rec.consecutive_alarms += 1
            logger.warning(
                "PH alarm | modality=%s trust=%.4f total_anomalies=%d",
                modality, rec.trust_score, rec.n_anomalies,
            )
            self._emit_event(TrustEvent(
                event_type="anomaly",
                modality=modality,
                trust_score=rec.trust_score,
                trust_level=rec.trust_level.value,
                message=f"CUSUM alarm #{rec.n_anomalies}",
            ))

            # Blacklist when cumulative anomaly count reaches threshold
            # (more robust than consecutive — catches erratic sensors too)
            if rec.n_anomalies >= self._config.anomaly_alarm_count:
                self._blacklist(rec)
                return False
        else:
            # Gradually recover consecutive alarm count when sensor behaves
            if rec.consecutive_alarms > 0 and consistency > 0.6:
                rec.consecutive_alarms = max(0, rec.consecutive_alarms - 1)

        # ── Emit level-change event ──────────────────────────────────
        if rec.trust_level != prev_level:
            self._emit_event(TrustEvent(
                event_type="level_change",
                modality=modality,
                trust_score=rec.trust_score,
                trust_level=rec.trust_level.value,
                message=f"Trust level: {prev_level.value} → {rec.trust_level.value}",
            ))

        return True

    def _blacklist(self, rec: SensorRecord) -> None:
        rec.is_blacklisted = True
        rec.blacklisted_at = time.time()
        rec.consecutive_alarms = 0
        logger.error(
            "SENSOR BLACKLISTED: %s | trust=%.4f | cooldown=%.0fs",
            rec.modality, rec.trust_score, self._config.blacklist_cooldown_seconds,
        )
        self._emit_event(TrustEvent(
            event_type="blacklisted",
            modality=rec.modality,
            trust_score=rec.trust_score,
            trust_level=TrustLevel.BLACKLISTED.value,
            message=(
                f"Sensor blacklisted after {self._config.anomaly_alarm_count} "
                f"consecutive anomalies.  Cooldown: "
                f"{self._config.blacklist_cooldown_seconds:.0f}s"
            ),
        ))

    def _reinstate(self, rec: SensorRecord) -> None:
        rec.is_blacklisted    = False
        rec.blacklisted_at    = None
        rec.consecutive_alarms = 0
        rec.ph_cumsum         = 0.0
        rec.ph_min            = 0.0
        # Reset trust score to neutral (1/φ) on reinstatement
        rec.trust_score       = GOLDEN_RATIO_RECIPROCAL
        logger.info("Sensor reinstated: %s", rec.modality)
        self._emit_event(TrustEvent(
            event_type="reinstated",
            modality=rec.modality,
            trust_score=rec.trust_score,
            trust_level=rec.trust_level.value,
            message="Sensor reinstated after cooldown.",
        ))

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def is_active(self, modality: str) -> bool:
        """Return True if the sensor is not blacklisted."""
        rec = self._sensors.get(modality)
        if rec is None:
            return True   # unseen → assume active
        if rec.is_blacklisted and rec.blacklisted_at is not None:
            cooldown = self._config.blacklist_cooldown_seconds
            if cooldown > 0 and time.time() - rec.blacklisted_at >= cooldown:
                self._reinstate(rec)
        return not rec.is_blacklisted

    def report(self, modality: str) -> Dict[str, Any]:
        """Legacy helper returning one sensor record."""
        return self._get_or_init(modality).to_dict()

    def trust_score(self, modality: str) -> float:
        """Return the current trust score for a modality (0–1)."""
        rec = self._sensors.get(modality)
        return rec.trust_score if rec else GOLDEN_RATIO_RECIPROCAL

    def trust_level(self, modality: str) -> TrustLevel:
        """Return the qualitative trust level for a modality."""
        rec = self._sensors.get(modality)
        return rec.trust_level if rec else TrustLevel.MODERATE

    def trust_report(self) -> Dict[str, Any]:
        """Return a full trust report for all sensors."""
        return {
            "sensors": {m: r.to_dict() for m, r in self._sensors.items()},
            "config":  self._config.to_dict(),
            "generated_at": time.time(),
        }

    def add_event_callback(self, callback: Callable[[TrustEvent], None]) -> None:
        """Register an additional event callback."""
        self._callbacks.append(callback)

    def force_reinstate(self, modality: str) -> None:
        """Manually reinstate a blacklisted sensor (operator override)."""
        rec = self._sensors.get(modality)
        if rec and rec.is_blacklisted:
            self._reinstate(rec)

    def reset(self, modality: Optional[str] = None) -> None:
        """Reset trust records."""
        if modality:
            self._sensors.pop(modality, None)
        else:
            self._sensors.clear()

    @property
    def active_modalities(self) -> List[str]:
        """List of modalities that are currently active (not blacklisted)."""
        return [m for m in self._sensors if self.is_active(m)]

    @property
    def blacklisted_modalities(self) -> List[str]:
        """List of currently blacklisted modalities."""
        return [m for m, r in self._sensors.items() if r.is_blacklisted]

    def __repr__(self) -> str:
        parts = [
            f"{m}={r.trust_score:.3f}({'BL' if r.is_blacklisted else r.trust_level.value[0].upper()})"
            for m, r in self._sensors.items()
        ]
        return f"SensorTrustScorer([{', '.join(parts)}])"
