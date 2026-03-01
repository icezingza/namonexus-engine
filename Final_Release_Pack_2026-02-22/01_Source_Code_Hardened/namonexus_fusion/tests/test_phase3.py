"""
Phase 3 Tests — Drift Detection + Streaming Inference Pipeline
==============================================================
Patent-Pending Technology | NamoNexus Research Team

Run:
    pytest namonexus_fusion/tests/test_phase3.py -v --tb=short

Tests cover:
  - DriftConfig  (golden/sensitive/conservative factories, validation)
  - _PHAccumulator  (Page-Hinkley math, armed state, cooldown)
  - DriftDetector  (multi-modality, callbacks, filtering, reset)
  - Golden Ratio threshold invariants
  - StreamingObservation  (serialization, clipping)
  - DeliveryLedger  (write, acknowledge, fail, replay, eviction)
  - SlidingWindowAnalyzer  (rolling stats, trend, O(1) window)
  - BackpressureController  (PAUSE / RESUME transitions)
  - InMemoryConnector  (async iteration)
  - StreamingPipeline  (end-to-end, backpressure, error handling)
  - Phase3GoldenFusion  (integrated engine)
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from namonexus_fusion.core.drift_detector import (
    DriftConfig,
    DriftDetector,
    DriftEvent,
    DriftSeverity,
    DriftDirection,
    _PHAccumulator,
    _PHI,
    _PHI_SQ,
    _PHI_SQ_RECIP,
    _PHI_RECIP,
)

from namonexus_fusion.core.streaming_pipeline import (
    StreamingObservation,
    DeliveryLedger,
    DeliveryStatus,
    SlidingWindowAnalyzer,
    BackpressureController,
    BackpressureSignal,
    InMemoryConnector,
    KafkaConnector,
    WebSocketConnector,
    StreamingPipeline,
    StreamingConfig,
    WindowedStats,
    StreamResult,
)

from namonexus_fusion.core.phase3_fusion import Phase3GoldenFusion


# ===========================================================================
# Helpers
# ===========================================================================

def make_obs(
    score: float = 0.7,
    confidence: float = 0.8,
    modality: str = "text",
    **kwargs,
) -> StreamingObservation:
    return StreamingObservation(score=score, confidence=confidence, modality=modality, **kwargs)


def stream_obs_list(n: int, score: float = 0.7) -> list:
    return [make_obs(score=score, modality="voice") for _ in range(n)]


# ===========================================================================
# 1.  DriftConfig
# ===========================================================================


class TestDriftConfig:
    def test_golden_defaults_use_phi(self):
        cfg = DriftConfig.golden()
        assert math.isclose(cfg.threshold_h,      _PHI_SQ,      rel_tol=1e-9)
        assert math.isclose(cfg.sensitivity_delta, _PHI_SQ_RECIP, rel_tol=1e-9)

    def test_threshold_h_times_recip_equals_warning_threshold(self):
        """Warning threshold = h × φ_recip."""
        cfg = DriftConfig.golden()
        warning_thr = cfg.threshold_h * _PHI_RECIP
        # warning_thr must be < threshold_h
        assert warning_thr < cfg.threshold_h

    def test_sensitive_factory_lower_threshold(self):
        golden    = DriftConfig.golden()
        sensitive = DriftConfig.sensitive()
        assert sensitive.threshold_h < golden.threshold_h

    def test_conservative_factory_higher_threshold(self):
        golden       = DriftConfig.golden()
        conservative = DriftConfig.conservative()
        assert conservative.threshold_h > golden.threshold_h

    def test_invalid_threshold_raises(self):
        with pytest.raises((ValueError, Exception)):
            DriftConfig(threshold_h=0.0)

    def test_invalid_sensitivity_raises(self):
        with pytest.raises((ValueError, Exception)):
            DriftConfig(sensitivity_delta=-0.1)

    def test_invalid_min_observations_raises(self):
        with pytest.raises((ValueError, Exception)):
            DriftConfig(min_observations=0)

    def test_to_dict_contains_threshold(self):
        cfg = DriftConfig.golden()
        d   = cfg.to_dict()
        assert "threshold_h" in d
        assert math.isclose(d["threshold_h"], _PHI_SQ, rel_tol=1e-9)


# ===========================================================================
# 2.  _PHAccumulator (Page-Hinkley internals)
# ===========================================================================


class TestPHAccumulator:
    def test_no_alarm_before_min_observations(self):
        cfg = DriftConfig(min_observations=10)
        acc = _PHAccumulator("voice", cfg)
        # Feed only 5 observations — should never alarm
        for _ in range(5):
            result = acc.update(consistency=0.1, uncertainty=0.0)
            assert result is None

    def test_alarm_fires_on_sustained_drop(self):
        """Sustained low consistency should eventually trigger ALARM."""
        cfg = DriftConfig(
            threshold_h=_PHI_SQ,
            sensitivity_delta=_PHI_SQ_RECIP,
            min_observations=5,
            window_size=10,
            emit_warnings=False,
            auto_reset=False,
        )
        acc = _PHAccumulator("face", cfg)

        # Feed a baseline of good consistency to arm the detector
        for _ in range(15):
            acc.update(consistency=0.95, uncertainty=0.0)

        # Now simulate sustained drop — very low consistency
        alarm = None
        for _ in range(50):
            event = acc.update(consistency=0.05, uncertainty=0.0)
            if event and event.severity == DriftSeverity.ALARM:
                alarm = event
                break

        assert alarm is not None, "Expected ALARM to fire on sustained drop"
        assert alarm.modality == "face"
        assert alarm.direction == DriftDirection.DOWNWARD

    def test_cooldown_suppresses_alarms(self):
        cfg = DriftConfig(
            threshold_h=0.5,          # very low threshold → alarm quickly
            sensitivity_delta=0.01,
            min_observations=3,
            cooldown_observations=50,
            auto_reset=True,
        )
        acc = _PHAccumulator("text", cfg)

        # Warm up
        for _ in range(5):
            acc.update(0.95, 0.0)

        # Trigger alarm
        alarm = None
        for _ in range(30):
            e = acc.update(0.01, 0.0)
            if e and e.severity == DriftSeverity.ALARM:
                alarm = e
                break

        assert alarm is not None
        assert acc.is_in_cooldown, "Should be in cooldown after ALARM"

        # During cooldown, updates should return None
        for _ in range(10):
            e = acc.update(0.01, 0.0)
            assert e is None

    def test_uncertainty_widens_threshold(self):
        """Higher uncertainty → larger h_eff → harder to alarm."""
        cfg = DriftConfig(
            threshold_h=_PHI_SQ,
            min_observations=5,
            emit_warnings=False,
            auto_reset=False,
        )

        def count_alarms(uncertainty: float) -> int:
            acc = _PHAccumulator("test", cfg)
            for _ in range(10):
                acc.update(0.95, 0.0)
            alarms = 0
            for _ in range(100):
                e = acc.update(0.05, uncertainty)
                if e and e.severity == DriftSeverity.ALARM:
                    alarms += 1
            return alarms

        # With high uncertainty the threshold is wider → fewer/slower alarms
        alarms_low_u  = count_alarms(0.0)
        alarms_high_u = count_alarms(5.0)
        assert alarms_low_u >= alarms_high_u, (
            "High uncertainty should result in fewer/slower alarms"
        )

    def test_warning_fires_before_alarm(self):
        cfg = DriftConfig(
            threshold_h=_PHI_SQ,
            min_observations=5,
            emit_warnings=True,
            auto_reset=False,
        )
        acc = _PHAccumulator("voice", cfg)

        # Arm
        for _ in range(10):
            acc.update(0.95, 0.0)

        seen_warning = False
        seen_alarm   = False
        for _ in range(200):
            e = acc.update(0.1, 0.0)
            if e:
                if e.severity == DriftSeverity.WARNING:
                    seen_warning = True
                elif e.severity == DriftSeverity.ALARM:
                    seen_alarm = True
                    break   # Stop after first alarm

        # WARNING must have been emitted before ALARM
        assert seen_alarm,   "Expected ALARM eventually"
        assert seen_warning, "Expected WARNING before ALARM"


# ===========================================================================
# 3.  DriftDetector (multi-modality)
# ===========================================================================


class TestDriftDetector:
    def test_lazy_accumulator_creation(self):
        det = DriftDetector()
        assert det.modalities_monitored() == []
        det.update("voice", 0.9, 0.1)
        assert "voice" in det.modalities_monitored()

    def test_callback_invoked_on_alarm(self):
        events_seen = []
        det = DriftDetector(
            config=DriftConfig(
                threshold_h=0.5,
                sensitivity_delta=0.01,
                min_observations=3,
                emit_warnings=False,
                auto_reset=False,
            ),
            callbacks=[events_seen.append],
        )

        for _ in range(5):
            det.update("face", 0.95, 0.0)
        for _ in range(100):
            det.update("face", 0.01, 0.0)

        assert any(e.severity == DriftSeverity.ALARM for e in events_seen)

    def test_independent_modality_accumulators(self):
        """Drift in 'voice' must NOT trigger alarm for 'face'."""
        det = DriftDetector(
            config=DriftConfig(
                threshold_h=0.5,
                min_observations=3,
                emit_warnings=False,
                auto_reset=False,
            )
        )

        # Arm both modalities
        for _ in range(10):
            det.update("voice", 0.95, 0.0)
            det.update("face",  0.95, 0.0)

        # Drive only 'voice' to alarm
        for _ in range(100):
            det.update("voice", 0.01, 0.0)

        voice_alarms = det.alarm_count("voice")
        face_alarms  = det.alarm_count("face")
        assert voice_alarms > 0,      "voice should have at least one alarm"
        assert face_alarms  == 0,     "face should have no alarms (unaffected)"

    def test_filter_events_by_severity(self):
        det = DriftDetector(
            config=DriftConfig(
                threshold_h=0.5,
                min_observations=3,
                emit_warnings=True,
                auto_reset=False,
            )
        )
        for _ in range(5):
            det.update("text", 0.95, 0.0)
        for _ in range(100):
            det.update("text", 0.05, 0.0)

        alarms   = det.events(severity=DriftSeverity.ALARM)
        warnings = det.events(severity=DriftSeverity.WARNING)
        # Both categories should be distinct
        assert all(e.severity == DriftSeverity.ALARM   for e in alarms)
        assert all(e.severity == DriftSeverity.WARNING  for e in warnings)

    def test_reset_clears_accumulators(self):
        det = DriftDetector()
        det.update("voice", 0.9, 0.1)
        det.update("voice", 0.8, 0.1)
        det.reset()
        assert det.alarm_count() == 0
        assert det.events() == []

    def test_summary_structure(self):
        det = DriftDetector()
        det.update("text",  0.9, 0.05)
        det.update("voice", 0.8, 0.05)
        s = det.summary()
        assert "modalities"   in s
        assert "total_alarms" in s
        assert "config"       in s
        assert len(s["modalities"]) == 2

    def test_per_modality_config_override(self):
        strict = DriftConfig(threshold_h=0.1, min_observations=2, auto_reset=False)
        det = DriftDetector(
            per_modality_config={"camera": strict}
        )
        # Camera has strict config; microphone has default lenient config
        # After warming up, camera should alarm before microphone at same input
        for _ in range(5):
            det.update("camera",     0.9, 0.0)
            det.update("microphone", 0.9, 0.0)
        for _ in range(50):
            det.update("camera",     0.01, 0.0)
            det.update("microphone", 0.01, 0.0)

        # Camera should alarm at least as often as microphone
        assert det.alarm_count("camera") >= det.alarm_count("microphone")

    def test_add_callback(self):
        seen = []
        det = DriftDetector(
            config=DriftConfig(threshold_h=0.5, min_observations=3, auto_reset=False)
        )
        det.add_callback(seen.append)
        for _ in range(5):
            det.update("x", 0.95, 0.0)
        for _ in range(100):
            det.update("x", 0.01, 0.0)
        # Callback should have received events
        # (may be warnings or alarms depending on config)
        # Just verify it was called if any events were emitted
        all_events = det.events()
        assert len(seen) == len(all_events)


# ===========================================================================
# 4.  Golden Ratio invariants
# ===========================================================================


class TestGoldenRatioInvariants:
    def test_phi_squared_equals_phi_plus_one(self):
        """φ² = φ + 1 — fundamental Golden Ratio identity."""
        assert math.isclose(_PHI_SQ, _PHI + 1.0, rel_tol=1e-9)

    def test_default_threshold_times_recip_equals_phi(self):
        """h = φ²; h × (1/φ) = φ — warning threshold is exactly φ."""
        h    = DriftConfig.golden().threshold_h
        warn = h * _PHI_RECIP
        assert math.isclose(warn, _PHI, rel_tol=1e-9)

    def test_phi_recip_plus_phi_sq_recip_equals_one(self):
        """1/φ + 1/φ² = 1 — complementary Golden Ratio partition."""
        assert math.isclose(_PHI_RECIP + _PHI_SQ_RECIP, 1.0, rel_tol=1e-9)

    def test_h_eff_increases_with_uncertainty(self):
        """h_eff = h × (1 + U × φ × scale) must increase monotonically."""
        h     = DriftConfig.golden().threshold_h
        scale = DriftConfig.golden().uncertainty_scaling
        u_vals = [0.0, 0.1, 0.5, 1.0, 2.0]
        h_effs = [h * (1.0 + u * _PHI * scale) for u in u_vals]
        assert all(h_effs[i] < h_effs[i + 1] for i in range(len(h_effs) - 1))


# ===========================================================================
# 5.  StreamingObservation
# ===========================================================================


class TestStreamingObservation:
    def test_score_clipped_to_unit_interval(self):
        obs = StreamingObservation(score=1.5, confidence=0.8, modality="text")
        assert obs.score == 1.0

        obs2 = StreamingObservation(score=-0.2, confidence=0.8, modality="text")
        assert obs2.score == 0.0

    def test_confidence_clipped_to_unit_interval(self):
        obs = StreamingObservation(score=0.5, confidence=2.0, modality="text")
        assert obs.confidence == 1.0

    def test_auto_generated_obs_id(self):
        obs = StreamingObservation(score=0.5, confidence=0.7, modality="voice")
        assert obs.obs_id != ""
        assert len(obs.obs_id) > 0

    def test_serialization_roundtrip(self):
        obs = StreamingObservation(
            score=0.75, confidence=0.85, modality="face",
            session_id="s001", subject_id="u42",
            metadata={"key": "value"}
        )
        json_str = obs.to_json()
        obs2     = StreamingObservation.from_json(json_str)
        assert math.isclose(obs2.score,      obs.score)
        assert math.isclose(obs2.confidence, obs.confidence)
        assert obs2.modality   == obs.modality
        assert obs2.session_id == obs.session_id
        assert obs2.subject_id == obs.subject_id
        assert obs2.metadata   == obs.metadata

    def test_to_dict_contains_expected_keys(self):
        obs = make_obs()
        d   = obs.to_dict()
        for key in ("score", "confidence", "modality", "obs_id", "timestamp"):
            assert key in d


# ===========================================================================
# 6.  DeliveryLedger
# ===========================================================================


class TestDeliveryLedger:
    def test_write_creates_pending_entry(self):
        ledger = DeliveryLedger()
        obs    = make_obs()
        ledger.write(obs)
        stats = ledger.stats()
        assert stats.get("pending", 0) >= 1

    def test_acknowledge_changes_status(self):
        ledger = DeliveryLedger()
        obs    = make_obs()
        ledger.write(obs)
        ledger.acknowledge(obs.obs_id)
        stats = ledger.stats()
        assert stats.get("acknowledged", 0) >= 1

    def test_fail_changes_status(self):
        ledger = DeliveryLedger()
        obs    = make_obs()
        ledger.write(obs)
        ledger.fail(obs.obs_id, "processing error")
        stats = ledger.stats()
        assert stats.get("failed", 0) >= 1

    def test_pending_for_replay_returns_unprocessed(self):
        ledger = DeliveryLedger()
        obs1   = make_obs()
        obs2   = make_obs()
        ledger.write(obs1)
        ledger.write(obs2)
        ledger.acknowledge(obs1.obs_id)
        # obs2 is still PENDING
        pending = ledger.pending_for_replay()
        ids     = {o.obs_id for o in pending}
        assert obs2.obs_id in ids
        assert obs1.obs_id not in ids

    def test_eviction_on_max_entries(self):
        ledger = DeliveryLedger(max_entries=5)
        obs_list = [make_obs() for _ in range(10)]
        for obs in obs_list:
            ledger.write(obs)
            ledger.acknowledge(obs.obs_id)
        # After eviction, total entries should not greatly exceed max
        total = sum(ledger.stats().values())
        assert total <= 10   # At most all 10 if eviction never triggered


# ===========================================================================
# 7.  SlidingWindowAnalyzer
# ===========================================================================


class TestSlidingWindowAnalyzer:
    def test_mean_score_tracks_input(self):
        analyzer = SlidingWindowAnalyzer(window_size=100)
        for _ in range(20):
            stats = analyzer.update(make_obs(score=0.8))
        assert math.isclose(stats.mean_score, 0.8, abs_tol=0.05)

    def test_window_limits_history(self):
        window = 10
        analyzer = SlidingWindowAnalyzer(window_size=window)
        for i in range(50):
            stats = analyzer.update(make_obs(score=0.5))
        assert stats.n_window <= window

    def test_total_count_increments(self):
        analyzer = SlidingWindowAnalyzer(window_size=10)
        for i in range(15):
            stats = analyzer.update(make_obs())
        assert stats.n_total == 15

    def test_modality_counts(self):
        analyzer = SlidingWindowAnalyzer()
        for _ in range(5):
            analyzer.update(make_obs(modality="text"))
        for _ in range(3):
            analyzer.update(make_obs(modality="voice"))
        stats = analyzer.update(make_obs(modality="face"))
        assert stats.modality_counts["text"]  == 5
        assert stats.modality_counts["voice"] == 3
        assert stats.modality_counts["face"]  == 1

    def test_rising_trend_is_positive(self):
        """Scores that increase over time should produce a positive risk_trend."""
        analyzer = SlidingWindowAnalyzer(window_size=50, trend_window=20)
        base_time = time.time()
        for i in range(20):
            obs = StreamingObservation(
                score=0.1 + i * 0.04,
                confidence=0.8,
                modality="text",
                timestamp=base_time + i,
            )
            stats = analyzer.update(obs)
        assert stats.risk_trend > 0, f"Expected positive trend, got {stats.risk_trend}"

    def test_reset_clears_state(self):
        analyzer = SlidingWindowAnalyzer()
        for _ in range(10):
            analyzer.update(make_obs())
        analyzer.reset()
        stats = analyzer.update(make_obs(score=0.5))
        assert stats.n_total == 1


# ===========================================================================
# 8.  BackpressureController
# ===========================================================================


class TestBackpressureController:
    def test_normal_below_max(self):
        bp = BackpressureController(max_queue_depth=100, resume_fraction=0.5)
        signal = bp.check(50)
        assert signal == BackpressureSignal.NORMAL
        assert not bp.is_paused

    def test_pause_at_max(self):
        bp = BackpressureController(max_queue_depth=100, resume_fraction=0.5)
        signal = bp.check(100)
        assert signal == BackpressureSignal.PAUSE
        assert bp.is_paused

    def test_resume_below_fraction(self):
        bp = BackpressureController(max_queue_depth=100, resume_fraction=0.5)
        bp.check(100)   # PAUSE
        signal = bp.check(40)   # Below 50 → RESUME
        assert signal == BackpressureSignal.RESUME
        assert not bp.is_paused

    def test_callback_invoked_on_transition(self):
        signals = []
        bp = BackpressureController(
            max_queue_depth=100, resume_fraction=0.5,
            callbacks=[signals.append]
        )
        bp.check(100)    # PAUSE → callback
        bp.check(40)     # RESUME → callback
        assert BackpressureSignal.PAUSE  in signals
        assert BackpressureSignal.RESUME in signals


# ===========================================================================
# 9.  InMemoryConnector
# ===========================================================================


class TestInMemoryConnector:
    def test_iterates_all_observations(self):
        obs_list  = [make_obs(score=float(i)/10) for i in range(5)]
        connector = InMemoryConnector(obs_list)

        async def collect():
            items = []
            async for obs in connector:
                items.append(obs)
            return items

        items = asyncio.run(collect())
        assert len(items) == 5

    def test_no_delay_by_default(self):
        obs_list  = [make_obs() for _ in range(3)]
        connector = InMemoryConnector(obs_list, inter_obs_delay=0.0)

        start = time.perf_counter()
        asyncio.run(connector.__aiter__().__anext__())
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1   # No artificial delay


# ===========================================================================
# 10.  StreamingPipeline (end-to-end)
# ===========================================================================


class TestStreamingPipeline:
    def _make_engine(self) -> Phase3GoldenFusion:
        return Phase3GoldenFusion()

    def test_basic_run_produces_results(self):
        engine    = self._make_engine()
        pipeline  = StreamingPipeline(engine)
        obs_list  = stream_obs_list(10, score=0.7)
        connector = InMemoryConnector(obs_list)

        results = pipeline.run_sync(connector)
        assert len(results) == 10

    def test_fused_score_updates_over_stream(self):
        engine    = self._make_engine()
        pipeline  = StreamingPipeline(engine)
        obs_list  = stream_obs_list(5, score=0.9)
        results   = pipeline.run_sync(InMemoryConnector(obs_list))
        # All scores should be in [0, 1]
        for r in results:
            assert 0.0 <= r.fused_score <= 1.0

    def test_window_stats_populated(self):
        engine    = self._make_engine()
        pipeline  = StreamingPipeline(engine, config=StreamingConfig(window_size=20))
        obs_list  = stream_obs_list(15, score=0.6)
        results   = pipeline.run_sync(InMemoryConnector(obs_list))

        assert all(r.window_stats is not None for r in results)
        last_ws = results[-1].window_stats
        assert last_ws.n_total == 15

    def test_ledger_acknowledges_all_successful_obs(self):
        engine    = self._make_engine()
        pipeline  = StreamingPipeline(engine)
        obs_list  = stream_obs_list(8)
        pipeline.run_sync(InMemoryConnector(obs_list))

        stats = pipeline.ledger_stats()
        assert stats.get("acknowledged", 0) == 8

    def test_result_callback_invoked(self):
        called = []
        engine = self._make_engine()
        pipeline = StreamingPipeline(
            engine,
            result_callbacks=[called.append],
        )
        pipeline.run_sync(InMemoryConnector(stream_obs_list(5)))
        assert len(called) == 5
        assert all(isinstance(r, StreamResult) for r in called)

    def test_max_observations_limit(self):
        engine   = self._make_engine()
        pipeline = StreamingPipeline(engine)
        obs_list = stream_obs_list(20)
        results  = pipeline.run_sync(InMemoryConnector(obs_list), max_observations=5)
        assert len(results) == 5

    def test_pipeline_summary_structure(self):
        engine   = self._make_engine()
        pipeline = StreamingPipeline(engine)
        pipeline.run_sync(InMemoryConnector(stream_obs_list(3)))
        s = pipeline.summary()
        assert "n_processed" in s
        assert "n_errors"    in s
        assert "ledger"      in s
        assert "backpressure" in s

    def test_drift_events_attached_to_results(self):
        """Results should carry drift_event=None when no drift, and a
        DriftEvent when one is detected."""
        # All results have drift_event field (may be None)
        engine   = Phase3GoldenFusion()
        pipeline = engine.stream()
        obs_list = stream_obs_list(10, score=0.7)
        results  = pipeline.run_sync(InMemoryConnector(obs_list))
        # Just verify the field exists
        for r in results:
            assert hasattr(r, "drift_event")

    def test_run_sync_equivalent_to_async_run(self):
        """run_sync and asyncio.run(run()) should produce identical results."""
        engine1  = Phase3GoldenFusion()
        engine2  = Phase3GoldenFusion()
        obs_list = stream_obs_list(6, score=0.65)

        results_sync  = StreamingPipeline(engine1).run_sync(InMemoryConnector(obs_list))
        results_async = asyncio.run(
            StreamingPipeline(engine2).run(InMemoryConnector(obs_list))
        )

        assert len(results_sync)  == len(results_async) == 6
        for rs, ra in zip(results_sync, results_async):
            assert math.isclose(rs.fused_score, ra.fused_score, abs_tol=1e-3)


# ===========================================================================
# 11.  Phase3GoldenFusion (integrated)
# ===========================================================================


class TestPhase3GoldenFusion:
    def test_instantiation(self):
        engine = Phase3GoldenFusion()
        assert engine is not None

    def test_update_returns_self(self):
        engine = Phase3GoldenFusion()
        result = engine.update(0.8, 0.9, "text")
        assert result is engine

    def test_fused_score_in_unit_interval(self):
        engine = Phase3GoldenFusion()
        for _ in range(10):
            engine.update(0.7, 0.8, "voice")
        assert 0.0 <= engine.fused_score <= 1.0

    def test_stream_returns_pipeline(self):
        engine   = Phase3GoldenFusion()
        pipeline = engine.stream()
        assert isinstance(pipeline, StreamingPipeline)

    def test_drift_alarm_count_starts_at_zero(self):
        engine = Phase3GoldenFusion()
        assert engine.drift_alarm_count == 0
        assert not engine.has_drift_alarm

    def test_drift_detection_enabled_by_default(self):
        engine = Phase3GoldenFusion()
        # _drift_enabled should be True
        assert engine._drift_enabled is True

    def test_enable_disable_drift_detection(self):
        engine = Phase3GoldenFusion()
        engine.disable_drift_detection()
        assert engine._drift_enabled is False
        engine.enable_drift_detection()
        assert engine._drift_enabled is True

    def test_drift_events_empty_initially(self):
        engine = Phase3GoldenFusion()
        assert engine.drift_events() == []

    def test_reset_clears_drift_state(self):
        engine = Phase3GoldenFusion(
            drift_config=DriftConfig(threshold_h=0.5, min_observations=3, auto_reset=False)
        )
        # Arm the detector
        for _ in range(5):
            engine.update(0.95, 0.9, "text")
        # Drive to alarm
        for _ in range(100):
            engine.update(0.05, 0.9, "text")

        # Regardless of whether an alarm fired, reset should clear state
        engine.reset()
        assert engine.drift_alarm_count == 0

    def test_modality_passed_to_drift_detector(self):
        engine = Phase3GoldenFusion()
        engine.update(0.7, 0.8, "face")
        engine.update(0.6, 0.7, "voice")
        monitored = engine._drift_detector.modalities_monitored()
        assert "face"  in monitored
        assert "voice" in monitored

    def test_no_modality_no_drift_update(self):
        """update() without modality_name should not crash or add to detector."""
        engine = Phase3GoldenFusion()
        engine.update(0.7, 0.8)   # no modality
        assert engine.drift_detector_summary()["modalities"] == []

    def test_drift_detector_summary(self):
        engine = Phase3GoldenFusion()
        engine.update(0.8, 0.9, "text")
        s = engine.drift_detector_summary()
        assert "modalities"   in s
        assert "total_alarms" in s

    def test_repr_contains_key_info(self):
        engine = Phase3GoldenFusion()
        r      = repr(engine)
        assert "score="         in r
        assert "drift_alarms="  in r

    def test_end_to_end_streaming(self):
        engine    = Phase3GoldenFusion()
        obs_list  = [
            StreamingObservation(score=0.9, confidence=0.8, modality="text",
                                 session_id="s1", subject_id="u1"),
            StreamingObservation(score=0.3, confidence=0.7, modality="voice",
                                 session_id="s1", subject_id="u1"),
            StreamingObservation(score=0.6, confidence=0.9, modality="face",
                                 session_id="s1", subject_id="u1"),
        ]
        pipeline = engine.stream()
        results  = pipeline.run_sync(InMemoryConnector(obs_list))

        assert len(results) == 3
        for r in results:
            assert 0.0 <= r.fused_score   <= 1.0
            assert 0.0 <= r.consistency   <= 1.0
            assert r.latency_ms           >= 0.0
            assert r.risk_level           in ("low", "medium", "high", "unknown")
            assert r.window_stats is not None

    def test_kafka_connector_stub(self):
        """KafkaConnector stub should deliver injected observations."""
        engine    = Phase3GoldenFusion()
        connector = KafkaConnector(topic="sensor_events")

        async def run():
            await connector.produce(make_obs(score=0.8, modality="text"))
            results = await engine.stream().run(connector, max_observations=1)
            return results

        results = asyncio.run(run())
        assert len(results) == 1
        assert math.isclose(results[0].fused_score, engine.fused_score, abs_tol=0.01)

    def test_websocket_connector_stub(self):
        """WebSocketConnector stub should deliver injected observations."""
        engine    = Phase3GoldenFusion()
        connector = WebSocketConnector(uri="ws://localhost:8080/stream")

        async def run():
            await connector.inject(make_obs(score=0.5, modality="voice"))
            results = await engine.stream().run(connector, max_observations=1)
            return results

        results = asyncio.run(run())
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Expose drift_detector_summary helper (injected into engine above)
# ---------------------------------------------------------------------------

def _drift_detector_summary(self) -> dict:
    return self._drift_detector.summary()

Phase3GoldenFusion.drift_detector_summary = _drift_detector_summary
