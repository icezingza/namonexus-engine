"""
Phase3GoldenFusion — Phase 3 Integrated Engine
===============================================
Patent-Pending Technology | NamoNexus Research Team

Integrates Features 3.1 and 3.2 into a single production-grade engine
that extends Phase2GoldenFusion (which itself extends Phase 1):

    Phase 1  TemporalGoldenFusion
        ↓ inherits
    Phase 2  Phase2GoldenFusion
        ↓ inherits
    Phase 3  Phase3GoldenFusion
        ├── DriftDetector        (3.1) — Page-Hinkley + φ control limits
        └── StreamingPipeline   (3.2) — backpressure, window, at-least-once

Architecture Decision
---------------------
Phase 3 is a **thin integration layer** — it wires DriftDetector into
the standard ``update()`` path and provides ``stream()`` as a first-class
method that returns a ready-made StreamingPipeline bound to this engine.

The design intentionally avoids subclassing the streaming logic: the
pipeline wraps the engine from the outside, preserving the clean
separation between fusion (state + math) and infrastructure (I/O +
delivery).

New Patent Claims Covered (Phase 3)
-------------------------------------
Claim 11: Drift Detection with φ-initialized Page-Hinkley + adaptive h_eff
Claim 12: Streaming inference with backpressure, sliding window, at-least-once

Usage
-----
::

    # Basic usage — drift detection on every update()
    engine = Phase3GoldenFusion()
    engine.update(0.85, 0.70, "text")
    engine.update(0.25, 0.90, "voice")

    if engine.has_drift_alarm:
        print("Drift detected!", engine.drift_events(severity=DriftSeverity.ALARM))

    # Streaming usage
    from namonexus_fusion.core.streaming_pipeline import (
        StreamingObservation, InMemoryConnector
    )

    observations = [
        StreamingObservation(score=0.8, confidence=0.9, modality="text"),
        StreamingObservation(score=0.3, confidence=0.7, modality="voice"),
    ]
    pipeline = engine.stream()
    results  = pipeline.run_sync(InMemoryConnector(observations))
    for r in results:
        print(r.fused_score, r.risk_level, r.window_stats.risk_trend)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .drift_detector import (
    DriftDetector,
    DriftConfig,
    DriftEvent,
    DriftSeverity,
)
from .streaming_pipeline import (
    StreamingPipeline,
    StreamingConfig,
    StreamingObservation,
    InMemoryConnector,
    KafkaConnector,
    WebSocketConnector,
    WindowedStats,
    StreamResult,
    DeliveryLedger,
    SlidingWindowAnalyzer,
)

logger = logging.getLogger(__name__)

# Attempt to import Phase 2 engine; fall back to a minimal stub for
# standalone testing without the full package installed.
try:
    from namonexus_fusion.core.phase2_fusion import Phase2GoldenFusion  # type: ignore
    _BASE_CLASS = Phase2GoldenFusion
    _BASE_NAME  = "Phase2GoldenFusion"
except ImportError:
    logger.warning(
        "Phase2GoldenFusion not found — Phase3GoldenFusion will use "
        "MinimalFusionStub as its base class.  Install the full package "
        "or place phase2/ on PYTHONPATH for production use."
    )

    class _MinimalFusionStub:
        """
        Minimal stub engine for standalone Phase 3 testing.

        Simulates a Beta-posterior fusion engine with Golden Ratio prior.
        Replace with Phase2GoldenFusion in production.
        """

        _PHI = (1.0 + 5.0 ** 0.5) / 2.0

        def __init__(self, **kwargs: Any) -> None:
            self._alpha = self._PHI * 2.0
            self._beta  = 2.0
            self._alpha0 = self._alpha
            self._beta0  = self._beta
            self._history: List[Dict] = []

        def update(
            self,
            score:         float,
            confidence:    float,
            modality_name: Optional[str] = None,
            **kwargs:      Any,
        ) -> "_MinimalFusionStub":
            n         = max(1.0, confidence * 10.0)
            successes = score * n
            failures  = n - successes
            self._alpha += successes
            self._beta  += failures
            self._history.append({
                "score": score, "confidence": confidence, "modality": modality_name
            })
            return self

        @property
        def fused_score(self) -> float:
            return self._alpha / (self._alpha + self._beta)

        @property
        def uncertainty(self) -> float:
            a, b = self._alpha, self._beta
            n    = a + b
            return float(np.sqrt(a * b / (n ** 2 * (n + 1)))) if n > 0 else 0.5

        @property
        def risk_level(self) -> str:
            s = self.fused_score
            if s >= 0.75:
                return "low"
            if s >= 0.45:
                return "medium"
            return "high"

        def reset(self) -> None:
            self._alpha = self._alpha0
            self._beta  = self._beta0
            self._history.clear()

        def get_state(self) -> Dict[str, Any]:
            return {
                "alpha": self._alpha,
                "beta":  self._beta,
                "config": {},
            }

    _BASE_CLASS = _MinimalFusionStub  # type: ignore
    _BASE_NAME  = "_MinimalFusionStub"


# ---------------------------------------------------------------------------
# Phase 3 integrated engine
# ---------------------------------------------------------------------------


class Phase3GoldenFusion(_BASE_CLASS):
    """
    Production-grade fusion engine with Drift Detection and Streaming.

    Extends Phase2GoldenFusion (Phase 1 + Phase 2) with:

    Feature 3.1 — DriftDetector:
        Runs a Page-Hinkley test on the per-modality consistency series.
        Fires WARNING and ALARM events when a sensor's behavior drifts
        from its historical baseline.  The detection threshold adapts
        to the engine's current posterior uncertainty via h_eff = h × (1 + U × φ).

    Feature 3.2 — StreamingPipeline:
        Exposes ``engine.stream()`` which returns a pre-configured
        StreamingPipeline bound to this engine.  Supports:
          • Backpressure (bounded queue depth)
          • Sliding-window statistics (mean score, risk trend, modality mix)
          • At-least-once delivery (DeliveryLedger write-ahead log)
          • Connector protocol (InMemory, Kafka, WebSocket adapters)

    Parameters
    ----------
    drift_config:
        DriftDetector configuration.  Pass ``None`` for Golden Ratio defaults.
    per_modality_drift_config:
        Optional per-modality DriftConfig overrides.
    streaming_config:
        StreamingPipeline configuration.
    drift_callbacks:
        Callables invoked on each DriftEvent:
        ``callback(event: DriftEvent) → None``.
    **base_kwargs:
        Forwarded to the base class (Phase2GoldenFusion or stub).
    """

    def __init__(
        self,
        drift_config:              Optional[DriftConfig]              = None,
        per_modality_drift_config: Optional[Dict[str, DriftConfig]]   = None,
        streaming_config:          Optional[StreamingConfig]          = None,
        drift_callbacks:           Optional[List[Callable[[DriftEvent], None]]] = None,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)

        self._drift_detector = DriftDetector(
            config              = drift_config,
            per_modality_config = per_modality_drift_config,
            callbacks           = drift_callbacks or [],
        )
        self._streaming_config = streaming_config or StreamingConfig()
        self._drift_enabled    = True

        logger.info(
            "Phase3GoldenFusion | base=%s drift_h=%.4f δ=%.4f queue=%d window=%d",
            _BASE_NAME,
            (drift_config or DriftConfig()).threshold_h,
            (drift_config or DriftConfig()).sensitivity_delta,
            self._streaming_config.max_queue_depth,
            self._streaming_config.window_size,
        )

    # ------------------------------------------------------------------
    # Overridden update — wires in drift detection
    # ------------------------------------------------------------------

    def update(
        self,
        score:         float,
        confidence:    float,
        modality_name: Optional[str] = None,
        metadata:      Optional[Dict[str, Any]] = None,
        **kwargs:      Any,
    ) -> "Phase3GoldenFusion":
        """
        Update posterior with optional drift detection.

        The consistency of this observation with the aggregate posterior
        is computed before the update (using the pre-update fused_score)
        and fed to the DriftDetector after.

        Parameters
        ----------
        score, confidence, modality_name, metadata:
            Same as Phase2GoldenFusion / base class.

        Returns
        -------
        Phase3GoldenFusion
            Self (fluent interface).
        """
        # Snapshot pre-update state for consistency
        score_before = getattr(self, "fused_score", 0.5)

        # Delegate to base class (Phase 1 + 2 logic)
        super().update(
            score         = score,
            confidence    = confidence,
            modality_name = modality_name,
            metadata      = metadata,
            **kwargs,
        )

        # Drift detection on consistency
        if self._drift_enabled and modality_name:
            consistency = 1.0 - abs(score - score_before)
            uncertainty = getattr(self, "uncertainty", 0.0)
            self._drift_detector.update(
                modality    = modality_name,
                consistency = consistency,
                uncertainty = float(uncertainty),
                metadata    = {"score": score, "confidence": confidence,
                               **(metadata or {})},
            )

        return self

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    def stream(
        self,
        streaming_config:  Optional[StreamingConfig]  = None,
        result_callbacks:  Optional[List[Callable]]   = None,
        error_callbacks:   Optional[List[Callable]]   = None,
    ) -> StreamingPipeline:
        """
        Create a ``StreamingPipeline`` bound to this engine.

        The pipeline shares the same DriftDetector — drift events
        detected during streaming are reflected in ``drift_events()``.

        Parameters
        ----------
        streaming_config:
            Override the engine-level config for this pipeline instance.
        result_callbacks:
            Async or sync callables invoked after each StreamResult.
        error_callbacks:
            Async or sync callables invoked on processing errors.

        Returns
        -------
        StreamingPipeline
            Ready to use: ``pipeline.run_sync(connector)`` or
            ``await pipeline.run(connector)``.

        Example
        -------
        ::

            engine   = Phase3GoldenFusion()
            pipeline = engine.stream()

            obs = [StreamingObservation(0.8, 0.9, "text"),
                   StreamingObservation(0.3, 0.7, "voice")]
            results = pipeline.run_sync(InMemoryConnector(obs))
        """
        cfg = streaming_config or self._streaming_config
        return StreamingPipeline(
            engine           = self,
            config           = cfg,
            drift_detector   = self._drift_detector,
            result_callbacks = result_callbacks,
            error_callbacks  = error_callbacks,
        )

    # ------------------------------------------------------------------
    # Drift diagnostics
    # ------------------------------------------------------------------

    @property
    def has_drift_alarm(self) -> bool:
        """True if at least one ALARM has been emitted since last reset."""
        return self._drift_detector.alarm_count() > 0

    def drift_events(
        self,
        modality:  Optional[str]          = None,
        severity:  Optional[DriftSeverity] = None,
        since:     Optional[float]         = None,
    ) -> List[DriftEvent]:
        """Return drift events, optionally filtered."""
        return self._drift_detector.events(
            modality=modality,
            severity=severity,
            since=since,
        )

    @property
    def drift_alarm_count(self) -> int:
        """Total ALARM events since last reset."""
        return self._drift_detector.alarm_count()

    def drift_summary(self) -> Dict[str, Any]:
        """Full diagnostic summary from the DriftDetector."""
        return self._drift_detector.summary()

    def enable_drift_detection(self) -> None:
        """Re-enable drift detection after ``disable_drift_detection()``."""
        self._drift_enabled = True

    def disable_drift_detection(self) -> None:
        """Temporarily disable drift detection (useful for calibration phases)."""
        self._drift_enabled = False

    def reset_drift(self, modality: Optional[str] = None) -> None:
        """
        Reset drift detector state.

        Parameters
        ----------
        modality:
            Reset only this modality; otherwise reset all.
        """
        self._drift_detector.reset(modality=modality)

    # ------------------------------------------------------------------
    # Overridden reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset base engine + drift detector."""
        super().reset()
        self._drift_detector.reset()

    # ------------------------------------------------------------------
    # Extended state
    # ------------------------------------------------------------------

    def get_state(self) -> Any:
        """Capture state including Phase 3 metadata."""
        state = super().get_state()
        if hasattr(state, "config"):
            state.config["drift"]     = self._drift_detector.summary()
            state.config["streaming"] = {
                "window_size":  self._streaming_config.window_size,
                "max_queue":    self._streaming_config.max_queue_depth,
            }
        return state

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        score    = getattr(self, "fused_score",  0.0)
        uncert   = getattr(self, "uncertainty",  0.0)
        risk_val = getattr(self, "risk_level",   "unknown")
        if callable(risk_val):
            risk_val = risk_val()
        n_alarms = self._drift_detector.alarm_count()
        return (
            f"Phase3GoldenFusion("
            f"score={score:.4f} ± {uncert:.4f}, "
            f"risk={risk_val}, "
            f"drift_alarms={n_alarms}, "
            f"streaming_window={self._streaming_config.window_size})"
        )
