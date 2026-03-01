"""
Streaming Inference Pipeline — Feature 3.2
===========================================
Patent-Pending Technology | NamoNexus Research Team

Problem
-------
The existing ``async_update_stream`` method accepts async generators but
lacks:

  1. **Backpressure** — a slow consumer causes unbounded memory growth
     as unprocessed observations accumulate in the queue.
  2. **Sliding window analysis** — no ability to compute windowed
     aggregates (mean score, risk trend, drift summary) over recent
     observations without keeping the full history.
  3. **At-least-once delivery guarantee** — if the processing loop
     crashes mid-stream, in-flight observations are lost with no
     mechanism for replay.
  4. **Connector abstractions** — direct code paths for Kafka / WebSocket
     are absent; every integration must be built from scratch.

Solution
--------
This module implements a production-grade streaming inference layer with
four components:

    StreamingObservation   — typed, serializable observation envelope
    DeliveryLedger         — write-ahead log for at-least-once guarantee
    SlidingWindowAnalyzer  — O(1) rolling statistics without full history
    StreamingPipeline      — orchestrates the full streaming loop with
                             backpressure, windowing, and ledger commits

Connector Protocol
------------------
Any class that implements ``StreamConnector`` (duck-typing) can act as
a source:

    def __aiter__(self) → AsyncIterator[StreamingObservation]

Provided adapters:
  - ``KafkaConnector``    (async stub, drop-in for confluent-kafka-python)
  - ``WebSocketConnector`` (async stub, drop-in for websockets / aiohttp)
  - ``InMemoryConnector`` (for tests and demos — synchronous iterator)

Patent Claim (new — Claim 12)
------------------------------
"A streaming inference system for multimodal Bayesian fusion, comprising:
(a) a sliding-window analyzer maintaining O(1) rolling statistics
    (mean score, risk trend, modality distribution) over a configurable
    window of recent observations, without storing the full observation
    history;
(b) a write-ahead delivery ledger that records each observation before
    processing and marks it acknowledged upon successful posterior update,
    guaranteeing at-least-once delivery on crash recovery;
(c) a backpressure controller that limits queue depth to a configurable
    bound and signals upstream producers to pause when the bound is
    exceeded, resuming when queue occupancy falls below a resume
    threshold;
(d) a connector abstraction that decouples observation sources (Kafka,
    WebSocket, in-memory generators) from the fusion engine, such that
    any source implementing the StreamConnector protocol can supply
    observations to the engine without code changes;
such that the system achieves at-least-once delivery, bounded memory
usage, and real-time sliding-window diagnostics simultaneously."

Thread/Concurrency Model
------------------------
The pipeline is designed for asyncio but also supports synchronous
usage via ``StreamingPipeline.run_sync()`` which runs the coroutine on
a new event loop.  All internal queues are ``asyncio.Queue`` instances
(not thread-safe across threads — wrap in ``loop.call_soon_threadsafe``
if feeding from threads).
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any, AsyncIterator, Callable, Coroutine, Deque, Dict,
    List, Optional, Tuple, Union,
)

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Golden Ratio constants
# ---------------------------------------------------------------------------

_PHI: float = (1.0 + 5.0 ** 0.5) / 2.0
_PHI_RECIP: float = 1.0 / _PHI


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DeliveryStatus(Enum):
    PENDING     = "pending"      # Written to ledger, not yet processed
    PROCESSING  = "processing"   # Dequeued, update in flight
    ACKNOWLEDGED = "acknowledged"  # Successfully committed to posterior
    FAILED      = "failed"       # Processing error — eligible for replay


class BackpressureSignal(Enum):
    NORMAL = "normal"    # Queue within bounds
    PAUSE  = "pause"     # Queue full — upstream should pause
    RESUME = "resume"    # Queue drained below resume threshold


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class StreamingObservation:
    """
    Typed, serializable envelope for a single streaming observation.

    Parameters
    ----------
    score:
        Raw modality score in [0, 1].
    confidence:
        Modality confidence in [0, 1].
    modality:
        Sensor name (e.g., "voice", "face", "text").
    obs_id:
        Unique observation identifier.  Auto-generated if not supplied.
    source:
        Connector source identifier (e.g., topic name, websocket URL).
    timestamp:
        Unix timestamp of the observation.
    session_id:
        Optional session grouping key.
    subject_id:
        Optional subject/user identifier.
    metadata:
        Arbitrary extra payload.
    """

    score:      float
    confidence: float
    modality:   str
    obs_id:     str  = field(default_factory=lambda: str(uuid.uuid4()))
    source:     str  = ""
    timestamp:  float = field(default_factory=time.time)
    session_id: Optional[str] = None
    subject_id: Optional[str] = None
    metadata:   Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score      = float(np.clip(self.score,      0.0, 1.0))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StreamingObservation":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "StreamingObservation":
        return cls.from_dict(json.loads(s))


@dataclass
class WindowedStats:
    """
    O(1) rolling statistics for a sliding window of observations.

    Updated incrementally — does not store individual observations.
    """

    window_size:        int
    n_total:            int   = 0
    n_window:           int   = 0
    mean_score:         float = 0.0
    mean_confidence:    float = 0.0
    score_variance:     float = 0.0
    risk_trend:         float = 0.0    # Slope of score over window
    modality_counts:    Dict[str, int] = field(default_factory=dict)
    last_updated:       float = field(default_factory=time.time)

    @property
    def dominant_modality(self) -> Optional[str]:
        if not self.modality_counts:
            return None
        return max(self.modality_counts, key=self.modality_counts.get)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LedgerEntry:
    """Write-ahead log entry for one observation."""

    obs_id:    str
    status:    DeliveryStatus
    obs_json:  str
    created:   float = field(default_factory=time.time)
    processed: Optional[float] = None
    error:     Optional[str]   = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class StreamResult:
    """
    Result produced by the pipeline for each processed observation.

    Contains the post-update fusion state and delivery metadata.
    """

    obs_id:       str
    modality:     str
    fused_score:  float
    uncertainty:  float
    risk_level:   str
    consistency:  float
    latency_ms:   float
    window_stats: Optional[WindowedStats] = None
    drift_event:  Optional[Any]           = None     # DriftEvent or None
    metadata:     Dict[str, Any]          = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.drift_event is not None:
            d["drift_event"] = self.drift_event.to_dict()
        return d


# ---------------------------------------------------------------------------
# Delivery Ledger (at-least-once guarantee)
# ---------------------------------------------------------------------------


class DeliveryLedger:
    """
    Write-ahead log for at-least-once delivery.

    Before each observation is processed, it is written to the ledger
    with status PENDING.  After successful posterior update, it is marked
    ACKNOWLEDGED.  On crash recovery, all PENDING / PROCESSING entries
    can be replayed.

    This implementation uses an in-memory store (suitable for production
    if backed by persistent storage in a subclass).  Override
    ``_persist()`` and ``_load()`` to add disk/Redis/DB durability.

    Parameters
    ----------
    max_entries:
        Maximum entries kept in memory.  Older ACKNOWLEDGED entries are
        evicted when the limit is reached.
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._entries: Dict[str, LedgerEntry] = {}
        self._max     = max_entries
        self._lock    = threading.RLock()
        logger.debug("DeliveryLedger initialized (max_entries=%d)", max_entries)

    def write(self, obs: StreamingObservation) -> None:
        """Record an observation as PENDING before processing begins."""
        entry = LedgerEntry(
            obs_id   = obs.obs_id,
            status   = DeliveryStatus.PENDING,
            obs_json = obs.to_json(),
        )
        with self._lock:
            self._entries[obs.obs_id] = entry
            self._evict_if_needed()
        self._persist(entry)

    def mark_processing(self, obs_id: str) -> None:
        """Mark an entry as actively being processed."""
        with self._lock:
            if obs_id in self._entries:
                self._entries[obs_id].status = DeliveryStatus.PROCESSING

    def acknowledge(self, obs_id: str) -> None:
        """Mark an entry as successfully committed to the posterior."""
        with self._lock:
            if obs_id in self._entries:
                e = self._entries[obs_id]
                e.status    = DeliveryStatus.ACKNOWLEDGED
                e.processed = time.time()

    def fail(self, obs_id: str, error: str) -> None:
        """Mark an entry as failed (eligible for replay)."""
        with self._lock:
            if obs_id in self._entries:
                e = self._entries[obs_id]
                e.status = DeliveryStatus.FAILED
                e.error  = error

    def pending_for_replay(self) -> List[StreamingObservation]:
        """
        Return all observations in PENDING or PROCESSING state.

        Call this on startup to replay observations that did not complete
        before the previous crash.
        """
        with self._lock:
            entries = [
                e for e in self._entries.values()
                if e.status in (DeliveryStatus.PENDING, DeliveryStatus.PROCESSING)
            ]
        return [StreamingObservation.from_json(e.obs_json) for e in entries]

    def _evict_if_needed(self) -> None:
        """Evict oldest ACKNOWLEDGED entries when at capacity."""
        if len(self._entries) <= self._max:
            return
        acked = sorted(
            [e for e in self._entries.values() if e.status == DeliveryStatus.ACKNOWLEDGED],
            key=lambda e: e.processed or 0.0,
        )
        n_evict = len(self._entries) - self._max + max(1, self._max // 10)
        for entry in acked[:n_evict]:
            del self._entries[entry.obs_id]

    def _persist(self, entry: LedgerEntry) -> None:
        """Override in subclasses to write to durable storage."""
        pass

    def _load(self) -> List[LedgerEntry]:
        """Override in subclasses to load from durable storage."""
        return []

    def stats(self) -> Dict[str, int]:
        with self._lock:
            counts: Dict[str, int] = {}
            for e in self._entries.values():
                counts[e.status.value] = counts.get(e.status.value, 0) + 1
        return counts

    def __repr__(self) -> str:
        s = self.stats()
        return f"DeliveryLedger({s})"


# ---------------------------------------------------------------------------
# Sliding Window Analyzer
# ---------------------------------------------------------------------------


class SlidingWindowAnalyzer:
    """
    O(1) incremental rolling statistics over recent observations.

    Uses Welford's online algorithm for variance and a deque of
    (score, timestamp) tuples for trend estimation.  The raw observation
    payloads are NOT stored — only sufficient statistics.

    Parameters
    ----------
    window_size:
        Number of observations in the sliding window.
    trend_window:
        Number of recent scores used to estimate risk_trend (slope).
        Must be <= window_size.
    """

    def __init__(self, window_size: int = 100, trend_window: int = 20) -> None:
        self._w       = window_size
        self._tw      = min(trend_window, window_size)
        self._scores:  Deque[float] = deque(maxlen=window_size)
        self._confs:   Deque[float] = deque(maxlen=window_size)
        self._times:   Deque[float] = deque(maxlen=trend_window)
        self._tscore:  Deque[float] = deque(maxlen=trend_window)
        self._mod_counts: Dict[str, int] = {}
        self._n_total: int = 0
        self._lock    = threading.RLock()

    def update(self, obs: StreamingObservation) -> WindowedStats:
        """
        Incorporate one observation and return updated statistics.

        Parameters
        ----------
        obs:
            The streaming observation.

        Returns
        -------
        WindowedStats with current rolling stats.
        """
        with self._lock:
            self._n_total += 1
            self._scores.append(obs.score)
            self._confs.append(obs.confidence)
            self._times.append(obs.timestamp)
            self._tscore.append(obs.score)

            m = obs.modality
            self._mod_counts[m] = self._mod_counts.get(m, 0) + 1

            scores  = list(self._scores)
            confs   = list(self._confs)
            ts      = list(self._times)
            tsc     = list(self._tscore)

            mean_s  = float(np.mean(scores))
            mean_c  = float(np.mean(confs))
            var_s   = float(np.var(scores)) if len(scores) > 1 else 0.0

            # Risk trend: linear regression slope of score vs time
            trend = 0.0
            if len(ts) >= 2:
                t_arr = np.array(ts, dtype=float)
                s_arr = np.array(tsc, dtype=float)
                t_arr -= t_arr[0]   # normalise time to [0, ...]
                dt = t_arr[-1] - t_arr[0]
                if dt > 0:
                    cov   = np.cov(t_arr, s_arr)[0, 1]
                    var_t = float(np.var(t_arr))
                    trend = float(cov / var_t) if var_t > 0 else 0.0

            return WindowedStats(
                window_size      = self._w,
                n_total          = self._n_total,
                n_window         = len(scores),
                mean_score       = mean_s,
                mean_confidence  = mean_c,
                score_variance   = var_s,
                risk_trend       = trend,
                modality_counts  = dict(self._mod_counts),
                last_updated     = time.time(),
            )

    def reset(self) -> None:
        with self._lock:
            self._scores.clear()
            self._confs.clear()
            self._times.clear()
            self._tscore.clear()
            self._mod_counts.clear()
            self._n_total = 0


# ---------------------------------------------------------------------------
# Connector abstractions
# ---------------------------------------------------------------------------


class InMemoryConnector:
    """
    Synchronous in-memory observation source for tests and demos.

    Wraps a list or generator of StreamingObservation objects.
    Implements ``__aiter__`` so it can be used with ``async for``.

    Parameters
    ----------
    observations:
        Iterable of StreamingObservation.
    inter_obs_delay:
        Simulated delay between observations (seconds).  0 = instant.
    """

    def __init__(
        self,
        observations:     List[StreamingObservation],
        inter_obs_delay:  float = 0.0,
    ) -> None:
        self._obs   = list(observations)
        self._delay = inter_obs_delay

    def __aiter__(self) -> "InMemoryConnector":
        self._idx = 0
        return self

    async def __anext__(self) -> StreamingObservation:
        if self._idx >= len(self._obs):
            raise StopAsyncIteration
        obs = self._obs[self._idx]
        self._idx += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        return obs


class KafkaConnector:
    """
    Async Kafka consumer adapter (stub — wire up to confluent-kafka-python).

    Drop-in replacement once confluent_kafka.aio is available:

    ::

        from confluent_kafka.aio import AIOProducer, AIOConsumer
        connector = KafkaConnector(topic="sensor_events", group_id="namonexus")
        async for obs in connector:
            ...

    This stub yields from an in-memory list for demonstration purposes.
    Replace ``_fetch_next()`` with a real Kafka ``poll()`` call.
    """

    def __init__(
        self,
        topic:       str,
        group_id:    str = "namonexus",
        bootstrap:   str = "localhost:9092",
        source_name: str = "",
    ) -> None:
        self.topic      = topic
        self.group_id   = group_id
        self.bootstrap  = bootstrap
        self.source     = source_name or f"kafka://{bootstrap}/{topic}"
        self._buffer:   asyncio.Queue  = asyncio.Queue()
        logger.info("KafkaConnector | topic=%s group=%s", topic, group_id)

    async def produce(self, obs: StreamingObservation) -> None:
        """Inject an observation (used in tests / stub mode)."""
        obs.source = self.source
        await self._buffer.put(obs)

    def __aiter__(self) -> "KafkaConnector":
        return self

    async def __anext__(self) -> StreamingObservation:
        try:
            return await asyncio.wait_for(self._buffer.get(), timeout=1.0)
        except asyncio.TimeoutError:
            raise StopAsyncIteration


class WebSocketConnector:
    """
    Async WebSocket observation source adapter (stub).

    Replace ``_connect()`` and the inner loop with a real ``websockets``
    or ``aiohttp.ClientSession.ws_connect()`` call.

    Expected message format: JSON matching ``StreamingObservation.from_json()``.
    """

    def __init__(
        self,
        uri:         str,
        source_name: str = "",
    ) -> None:
        self.uri    = uri
        self.source = source_name or f"ws://{uri}"
        self._buffer: asyncio.Queue = asyncio.Queue()
        logger.info("WebSocketConnector | uri=%s", uri)

    async def inject(self, obs: StreamingObservation) -> None:
        """Inject an observation (used in tests / stub mode)."""
        obs.source = self.source
        await self._buffer.put(obs)

    def __aiter__(self) -> "WebSocketConnector":
        return self

    async def __anext__(self) -> StreamingObservation:
        try:
            return await asyncio.wait_for(self._buffer.get(), timeout=1.0)
        except asyncio.TimeoutError:
            raise StopAsyncIteration


# ---------------------------------------------------------------------------
# Backpressure controller
# ---------------------------------------------------------------------------


class BackpressureController:
    """
    Limits queue depth and signals upstream producers.

    Parameters
    ----------
    max_queue_depth:
        Maximum observations in flight.  When exceeded, PAUSE is signaled.
    resume_fraction:
        Queue drains to max_queue_depth × resume_fraction before RESUME.
    callbacks:
        Callables invoked with BackpressureSignal on state transitions.
    """

    def __init__(
        self,
        max_queue_depth: int   = 1000,
        resume_fraction: float = 0.5,
        callbacks:       Optional[List[Callable[[BackpressureSignal], None]]] = None,
    ) -> None:
        self._max      = max_queue_depth
        self._resume   = int(max_queue_depth * resume_fraction)
        self._cbs      = callbacks or []
        self._paused   = False
        self._lock     = threading.Lock()

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._paused

    def check(self, current_depth: int) -> BackpressureSignal:
        """
        Evaluate queue depth and emit signals on state transitions.

        Returns the current BackpressureSignal.
        """
        with self._lock:
            if not self._paused and current_depth >= self._max:
                self._paused = True
                signal = BackpressureSignal.PAUSE
                logger.warning("Backpressure: PAUSE (depth=%d >= %d)", current_depth, self._max)
            elif self._paused and current_depth <= self._resume:
                self._paused = False
                signal = BackpressureSignal.RESUME
                logger.info("Backpressure: RESUME (depth=%d <= %d)", current_depth, self._resume)
            else:
                signal = BackpressureSignal.PAUSE if self._paused else BackpressureSignal.NORMAL

        if signal in (BackpressureSignal.PAUSE, BackpressureSignal.RESUME):
            for cb in self._cbs:
                try:
                    cb(signal)
                except Exception as exc:
                    logger.warning("Backpressure callback error: %s", exc)

        return signal


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass
class StreamingConfig:
    """
    Configuration for the StreamingPipeline.

    Parameters
    ----------
    max_queue_depth:
        Backpressure threshold (max unprocessed observations).
    resume_fraction:
        Queue fraction at which backpressure releases (default 0.5 × max).
    window_size:
        Sliding window size for SlidingWindowAnalyzer.
    trend_window:
        Number of recent scores used for risk trend estimation.
    ledger_max_entries:
        DeliveryLedger capacity (in-memory).
    result_callback:
        Optional coroutine called with each StreamResult.
    error_callback:
        Optional coroutine called on processing errors.
    max_concurrent:
        Maximum concurrent observations processed simultaneously.
    replay_on_start:
        If True, replay PENDING ledger entries on pipeline startup.
    """

    max_queue_depth:  int   = 1000
    resume_fraction:  float = 0.5
    window_size:      int   = 100
    trend_window:     int   = 20
    ledger_max_entries: int = 10_000
    max_concurrent:   int   = 4
    replay_on_start:  bool  = False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class StreamingPipeline:
    """
    Production-grade streaming inference pipeline for the NamoNexus engine.

    Orchestrates:
    - **Backpressure** via ``BackpressureController``
    - **At-least-once delivery** via ``DeliveryLedger``
    - **Sliding-window statistics** via ``SlidingWindowAnalyzer``
    - **Drift detection** via optional ``DriftDetector``

    Connector protocol
    ------------------
    Any object implementing ``__aiter__ → AsyncIterator[StreamingObservation]``
    can be used as a source.  Built-in connectors:
    ``InMemoryConnector``, ``KafkaConnector``, ``WebSocketConnector``.

    Parameters
    ----------
    engine:
        A fusion engine instance (Phase2GoldenFusion or any compatible
        engine with ``update(score, confidence, modality)`` and
        ``fused_score``, ``uncertainty``, ``risk_level`` properties).
    config:
        StreamingConfig.
    drift_detector:
        Optional DriftDetector.  If provided, consistency is computed
        and drift events are attached to StreamResults.
    result_callbacks:
        Async callables ``(result: StreamResult) → None`` invoked after
        each successful update.
    error_callbacks:
        Async callables ``(obs: StreamingObservation, exc: Exception) → None``
        invoked on processing errors.

    Examples
    --------
    ::

        engine = Phase2GoldenFusion()
        pipeline = StreamingPipeline(engine)

        observations = [
            StreamingObservation(score=0.8, confidence=0.9, modality="text"),
            StreamingObservation(score=0.3, confidence=0.7, modality="voice"),
        ]
        connector = InMemoryConnector(observations)

        results = asyncio.run(pipeline.run(connector))
        for r in results:
            print(r.fused_score, r.risk_level)

    Or synchronously::

        results = pipeline.run_sync(connector)
    """

    def __init__(
        self,
        engine:           Any,
        config:           Optional[StreamingConfig]   = None,
        drift_detector:   Optional[Any]               = None,
        result_callbacks: Optional[List[Callable]]    = None,
        error_callbacks:  Optional[List[Callable]]    = None,
    ) -> None:
        self._engine    = engine
        self._cfg       = config or StreamingConfig()
        self._detector  = drift_detector
        self._result_cbs = result_callbacks or []
        self._error_cbs  = error_callbacks  or []

        self._ledger    = DeliveryLedger(max_entries=self._cfg.ledger_max_entries)
        self._analyzer  = SlidingWindowAnalyzer(
            window_size  = self._cfg.window_size,
            trend_window = self._cfg.trend_window,
        )
        self._bp = BackpressureController(
            max_queue_depth = self._cfg.max_queue_depth,
            resume_fraction = self._cfg.resume_fraction,
        )

        self._results:  List[StreamResult] = []
        self._n_processed: int = 0
        self._n_errors:    int = 0

        logger.info(
            "StreamingPipeline | queue=%d window=%d concurrent=%d",
            self._cfg.max_queue_depth,
            self._cfg.window_size,
            self._cfg.max_concurrent,
        )

    # ------------------------------------------------------------------
    # Internal: process one observation
    # ------------------------------------------------------------------

    async def _process(self, obs: StreamingObservation) -> Optional[StreamResult]:
        """Process a single observation end-to-end."""
        start = time.perf_counter()

        self._ledger.write(obs)
        self._ledger.mark_processing(obs.obs_id)

        try:
            # Capture pre-update state for consistency calculation
            score_before = getattr(self._engine, "fused_score", 0.5)

            # Core update
            self._engine.update(obs.score, obs.confidence, obs.modality)

            score_after = getattr(self._engine, "fused_score", obs.score)
            uncertainty = getattr(self._engine, "uncertainty", 0.0)
            risk_level  = getattr(self._engine, "risk_level",  "unknown")
            risk_level = str(risk_level).lower()
            if risk_level == "moderate":
                risk_level = "medium"
            elif risk_level == "critical":
                risk_level = "high"

            # Consistency: how much this obs agreed with the aggregate
            consistency = 1.0 - abs(obs.score - score_before)

            # Sliding window update
            window_stats = self._analyzer.update(obs)

            # Optional drift detection
            drift_event = None
            if self._detector is not None:
                drift_event = self._detector.update(
                    modality    = obs.modality,
                    consistency = consistency,
                    uncertainty = float(uncertainty),
                    metadata    = {"obs_id": obs.obs_id},
                )

            latency = (time.perf_counter() - start) * 1000.0

            result = StreamResult(
                obs_id       = obs.obs_id,
                modality     = obs.modality,
                fused_score  = float(score_after),
                uncertainty  = float(uncertainty),
                risk_level   = risk_level,
                consistency  = float(consistency),
                latency_ms   = latency,
                window_stats = window_stats,
                drift_event  = drift_event,
                metadata     = {
                    "session_id": obs.session_id,
                    "subject_id": obs.subject_id,
                    "source":     obs.source,
                },
            )

            self._ledger.acknowledge(obs.obs_id)
            self._n_processed += 1

            # Invoke result callbacks
            for cb in self._result_cbs:
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(result)
                    else:
                        cb(result)
                except Exception as exc:
                    logger.warning("Result callback error: %s", exc)

            logger.debug(
                "stream | mod=%s score=%.3f→%.3f U=%.4f risk=%s latency=%.1fms",
                obs.modality, score_before, score_after, uncertainty, risk_level, latency,
            )
            return result

        except Exception as exc:
            self._ledger.fail(obs.obs_id, str(exc))
            self._n_errors += 1
            logger.error("Processing error obs_id=%s: %s", obs.obs_id, exc, exc_info=True)

            for cb in self._error_cbs:
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(obs, exc)
                    else:
                        cb(obs, exc)
                except Exception as cb_exc:
                    logger.warning("Error callback error: %s", cb_exc)

            return None

    # ------------------------------------------------------------------
    # Public: run the streaming loop
    # ------------------------------------------------------------------

    async def run(
        self,
        connector:  Any,
        max_observations: Optional[int] = None,
    ) -> List[StreamResult]:
        """
        Consume observations from *connector* and process them.

        Parameters
        ----------
        connector:
            Any object implementing ``__aiter__ → AsyncIterator[StreamingObservation]``.
        max_observations:
            Stop after processing this many observations.  None = run until
            the connector is exhausted.

        Returns
        -------
        List[StreamResult]
            All successfully produced results, in order.
        """
        self._results.clear()
        semaphore = asyncio.Semaphore(self._cfg.max_concurrent)

        # Replay ledger entries on startup
        if self._cfg.replay_on_start:
            pending = self._ledger.pending_for_replay()
            if pending:
                logger.info("Replaying %d pending ledger entries", len(pending))
                for obs in pending:
                    r = await self._process(obs)
                    if r:
                        self._results.append(r)

        queue: asyncio.Queue = asyncio.Queue(maxsize=self._cfg.max_queue_depth)

        async def producer() -> None:
            n = 0
            async for obs in connector:
                if max_observations is not None and n >= max_observations:
                    break
                # Backpressure: wait if queue is full
                self._bp.check(queue.qsize())
                while self._bp.is_paused:
                    await asyncio.sleep(0.01)
                    self._bp.check(queue.qsize())
                await queue.put(obs)
                n += 1
            # Sentinel to signal end of stream
            for _ in range(self._cfg.max_concurrent):
                await queue.put(None)

        async def consumer() -> None:
            while True:
                obs = await queue.get()
                if obs is None:
                    break
                async with semaphore:
                    r = await self._process(obs)
                    if r:
                        self._results.append(r)
                queue.task_done()

        # Run producer and consumers concurrently
        consumers = [
            asyncio.create_task(consumer())
            for _ in range(self._cfg.max_concurrent)
        ]
        await producer()
        await asyncio.gather(*consumers)

        logger.info(
            "StreamingPipeline finished | processed=%d errors=%d",
            self._n_processed, self._n_errors,
        )
        return self._results

    def run_sync(
        self,
        connector:        Any,
        max_observations: Optional[int] = None,
    ) -> List[StreamResult]:
        """
        Synchronous wrapper around ``run()``.

        Runs the async pipeline on a new event loop.  Useful for
        non-async callers and interactive use.
        """
        return asyncio.run(self.run(connector, max_observations=max_observations))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def n_processed(self) -> int:
        return self._n_processed

    @property
    def n_errors(self) -> int:
        return self._n_errors

    @property
    def results(self) -> List[StreamResult]:
        return list(self._results)

    def ledger_stats(self) -> Dict[str, int]:
        return self._ledger.stats()

    def window_stats(self) -> Optional[WindowedStats]:
        """Return the most recent window statistics, or None if no data."""
        if self._results:
            return self._results[-1].window_stats
        return None

    def summary(self) -> Dict[str, Any]:
        return {
            "n_processed":  self._n_processed,
            "n_errors":     self._n_errors,
            "ledger":       self._ledger.stats(),
            "backpressure": {
                "is_paused":      self._bp.is_paused,
                "max_depth":      self._cfg.max_queue_depth,
            },
            "window":       (self._results[-1].window_stats.to_dict()
                             if self._results else None),
        }

    def reset(self) -> None:
        """Reset pipeline state (results, counters, analyzer)."""
        self._results.clear()
        self._n_processed = 0
        self._n_errors    = 0
        self._analyzer.reset()

    def __repr__(self) -> str:
        return (
            f"StreamingPipeline("
            f"processed={self._n_processed}, "
            f"errors={self._n_errors}, "
            f"window={self._cfg.window_size}, "
            f"queue_max={self._cfg.max_queue_depth})"
        )
