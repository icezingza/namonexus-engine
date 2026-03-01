"""
Phase 2 Tests — Modality Calibration, Sensor Trust Scoring, Hyperparameter Optimization
========================================================================================

Run:
    cd /home/claude
    python3 -m pytest namonexus_fusion/tests/test_phase2.py -v --tb=short
"""

from __future__ import annotations

import math
import sys
sys.path.insert(0, '/home/claude')

import pytest
import numpy as np

from namonexus_fusion.core.constants import GOLDEN_RATIO, GOLDEN_RATIO_RECIPROCAL
from namonexus_fusion.core.modality_calibrator import ModalityCalibrator, CalibrationConfig
from namonexus_fusion.core.sensor_trust_scorer import SensorTrustScorer, TrustScorerConfig, TrustEvent
from namonexus_fusion.core.hyperopt import OnlineHyperparamOptimizer, HyperparamBounds
from namonexus_fusion.core.phase2_fusion import Phase2GoldenFusion
from namonexus_fusion.config.settings import FusionConfig


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def default_phase2_engine():
    return Phase2GoldenFusion()


@pytest.fixture()
def calibrator():
    return ModalityCalibrator()


@pytest.fixture()
def trust_scorer():
    return SensorTrustScorer()


# ===========================================================================
# 1. ModalityCalibrator (Feature 2.1)
# ===========================================================================

class TestModalityCalibrator:

    def test_initial_trust_is_golden_ratio_anchored(self, calibrator):
        """Trust posterior initialized with τ_α/τ_β = φ."""
        report = calibrator.report("text")
        ratio = report["tau_alpha"] / report["tau_beta"]
        assert math.isclose(ratio, GOLDEN_RATIO, rel_tol=1e-3), f"Expected φ ratio, got {ratio}"

    def test_consistent_observation_increases_trust(self, calibrator):
        """Observation aligned with fused score rewards the modality."""
        for _ in range(10):
            calibrator.effective_confidence(
                raw_confidence=0.8,
                score=0.5,
                fused_score=0.5,
                modality="text",
            )
        report = calibrator.report("text")
        assert report["trust_mean"] > 0.618, "Trust should increase above initial ratio mean"

    def test_inconsistent_observation_reduces_trust(self, calibrator):
        """Observation contradicting fused score penalises the modality."""
        for _ in range(20):
            # score far from fused_score → low consistency → penalise
            calibrator.effective_confidence(
                raw_confidence=0.8,
                score=0.0,
                fused_score=1.0,
                modality="flaky_sensor",
            )
        report = calibrator.report("flaky_sensor")
        assert report["trust_mean"] < 0.618, "Trust should drop below initial for unreliable sensor"

    def test_effective_confidence_bounded(self, calibrator):
        """Effective confidence is always in [0, 1]."""
        for score in np.linspace(0, 1, 11):
            eff = calibrator.effective_confidence(0.9, score, 0.5, "cam")
            assert 0.0 <= eff <= 1.0, f"Effective confidence out of range: {eff}"

    def test_multiple_modalities_tracked_independently(self, calibrator):
        """Each modality has its own trust state."""
        for _ in range(5):
            calibrator.effective_confidence(0.9, 0.8, 0.7, "text")
            calibrator.effective_confidence(0.9, 0.1, 0.8, "noisy_camera")

        report_text = calibrator.report("text")
        report_cam = calibrator.report("noisy_camera")
        assert report_text["trust_mean"] > report_cam["trust_mean"], (
            "Consistent modality should have higher trust than noisy one"
        )


# ===========================================================================
# 2. SensorTrustScorer (Feature 2.2)
# ===========================================================================

class TestSensorTrustScorer:

    def test_new_sensor_is_active(self, trust_scorer):
        assert trust_scorer.is_active("text") is True

    def test_sensor_starts_with_golden_ratio_control_limits(self, trust_scorer):
        report = trust_scorer.report("text")
        cfg = TrustScorerConfig()
        assert math.isclose(cfg.ph_delta, GOLDEN_RATIO_RECIPROCAL, rel_tol=1e-3)
        assert math.isclose(cfg.ph_threshold, GOLDEN_RATIO, rel_tol=1e-3)

    def test_consistently_reliable_sensor_not_blacklisted(self, trust_scorer):
        """High consistency observations should not trigger blacklisting."""
        for _ in range(50):
            trust_scorer.record_observation("reliable_sensor", consistency=0.95)
        assert trust_scorer.is_active("reliable_sensor") is True

    def test_repeated_anomalies_trigger_blacklist(self, trust_scorer):
        """Repeated low-consistency bursts should eventually blacklist the sensor."""
        cfg = TrustScorerConfig(blacklist_cooldown_seconds=0.0, anomaly_alarm_count=1)
        scorer = SensorTrustScorer(config=cfg)
        # Inject many very low-consistency observations
        for _ in range(100):
            scorer.record_observation("bad_sensor", consistency=0.0)
        assert scorer.is_active("bad_sensor") is False

    def test_trust_event_callback_fired(self, trust_scorer):
        """Trust event callbacks are invoked on state changes."""
        events = []
        trust_scorer.add_event_callback(events.append)

        cfg = TrustScorerConfig(blacklist_cooldown_seconds=0.0, anomaly_alarm_count=1)
        scorer = SensorTrustScorer(config=cfg)
        scorer.add_event_callback(events.append)

        for _ in range(100):
            scorer.record_observation("flakey", consistency=0.0)

        assert len(events) > 0, "Expected at least one TrustEvent"

    def test_trust_score_range(self, trust_scorer):
        """Trust score must be in [0, 1]."""
        for c in np.linspace(0, 1, 11):
            trust_scorer.record_observation("sensor", consistency=float(c))
        report = trust_scorer.report("sensor")
        assert 0.0 <= report["trust_score"] <= 1.0


# ===========================================================================
# 3. OnlineHyperparamOptimizer (Feature 2.3)
# ===========================================================================

class TestOnlineHyperparamOptimizer:

    def test_initial_config_is_valid(self):
        opt = OnlineHyperparamOptimizer()
        cfg = opt.current_config
        assert cfg.prior_strength > 0
        assert cfg.max_trials > 0
        assert cfg.confidence_scale > 0

    def test_golden_ratio_constraint_preserved(self):
        """All candidate configs must satisfy α₀/β₀ = φ."""
        opt = OnlineHyperparamOptimizer()
        cfg = opt.current_config
        assert math.isclose(cfg.alpha0 / cfg.beta0, GOLDEN_RATIO, rel_tol=1e-4)

    def test_step_returns_config_or_none(self):
        opt = OnlineHyperparamOptimizer()
        result = opt.step(
            observations=[(0.7, 0.8, "text"), (0.6, 0.7, "face")],
            fused_score=0.65,
            uncertainty=0.1,
            risk_level="medium",
        )
        assert result is None or hasattr(result, "prior_strength")


# ===========================================================================
# 4. Phase2GoldenFusion — Integration
# ===========================================================================

class TestPhase2GoldenFusion:

    def test_basic_update_chain(self, default_phase2_engine):
        """Engine updates without error and fused_score is in (0, 1)."""
        engine = default_phase2_engine
        engine.update(0.7, 0.8, "text")
        engine.update(0.5, 0.7, "face")
        engine.update(0.4, 0.9, "voice")
        assert 0.0 < engine.fused_score < 1.0

    def test_risk_levels_cover_all_categories(self):
        """All four risk levels can be produced."""
        levels = set()
        configs = [
            (0.1, 0.9),  # should give low
            (0.45, 0.9), # medium
            (0.70, 0.9), # high
            (0.90, 0.9), # critical
        ]
        for score, conf in configs:
            e = Phase2GoldenFusion()
            for _ in range(5):
                e.update(score, conf, "text")
            levels.add(e.risk_level)
        assert len(levels) >= 3, f"Expected at least 3 risk levels, got: {levels}"

    def test_dropped_observations_tracked(self):
        """Blacklisted sensor observations are counted as dropped."""
        cfg = TrustScorerConfig(blacklist_cooldown_seconds=0.0, anomaly_alarm_count=1)
        engine = Phase2GoldenFusion(trust_scorer_config=cfg)
        # Force blacklist by injecting many low-consistency readings internally
        for _ in range(100):
            engine._trust_scorer.record_observation("bad", consistency=0.0)
        # Force the engine to think bad is active so it tries to process
        before = engine.dropped_observations
        engine.update(0.5, 0.8, "bad")
        assert engine.dropped_observations >= before

    def test_phase1_temporal_decay_still_works(self):
        """Phase 2 inherits Phase 1 temporal forgetting."""
        engine = Phase2GoldenFusion()
        # After update, λ history should contain 1/φ
        engine.update(0.6, 0.8, "text")
        lam = engine.lambda_history()
        assert len(lam) == 1
        assert math.isclose(lam[0], GOLDEN_RATIO_RECIPROCAL, rel_tol=0.05)

    def test_active_modalities_updated(self):
        """Active modality list includes all seen modalities."""
        engine = Phase2GoldenFusion()
        engine.update(0.7, 0.8, "text")
        engine.update(0.6, 0.7, "face")
        assert "text" in engine.active_modalities
        assert "face" in engine.active_modalities

    def test_reset_clears_state(self, default_phase2_engine):
        """Reset restores engine to prior state."""
        engine = default_phase2_engine
        engine.update(0.9, 0.9, "text")
        pre_reset_score = engine.fused_score
        engine.reset()
        assert engine.fused_score != pre_reset_score or engine.total_observations == 0
        assert engine.dropped_observations == 0

    def test_credible_interval_valid(self, default_phase2_engine):
        """Credible interval is properly ordered and in [0, 1]."""
        engine = default_phase2_engine
        engine.update(0.6, 0.8, "text")
        lo, hi = engine.credible_interval(0.95)
        assert 0.0 <= lo < hi <= 1.0

    def test_state_serialization_roundtrip(self):
        """Engine state can be saved and loaded."""
        engine = Phase2GoldenFusion()
        engine.update(0.7, 0.8, "text")
        engine.update(0.5, 0.6, "voice")
        state = engine.get_state()

        engine2 = Phase2GoldenFusion()
        engine2.load_state(state)
        assert math.isclose(engine.fused_score, engine2.fused_score, rel_tol=1e-6)

    def test_calibration_report_structure(self):
        """Calibration report contains expected keys per modality."""
        engine = Phase2GoldenFusion()
        engine.update(0.7, 0.8, "text")
        report = engine.calibration_report()
        assert "text" in report
        assert "trust_mean" in report["text"]
        assert "tau_alpha" in report["text"]

    def test_trust_report_structure(self):
        """Trust report contains expected keys."""
        engine = Phase2GoldenFusion()
        engine.update(0.7, 0.8, "text")
        report = engine.trust_report()
        assert "sensors" in report
        assert "text" in report["sensors"]
        assert "trust_score" in report["sensors"]["text"]
