"""
Phase 1 Tests — Temporal Bayesian Filtering + Empirical Prior Learning
======================================================================

Run:
    pytest namonexus_fusion/tests/test_phase1.py -v --tb=short
"""

from __future__ import annotations

import math
import time
import pytest
from hypothesis import given, settings, strategies as st

from namonexus_fusion.core.constants import GOLDEN_RATIO, GOLDEN_RATIO_RECIPROCAL
from namonexus_fusion.core.temporal_filter import (
    TemporalBayesianFilter, TemporalConfig
)
from namonexus_fusion.core.empirical_prior import (
    EmpiricalPriorLearner, PriorLearningConfig,
    SyntheticSessionGenerator, SubjectProfile, Session, ModalityObservation
)
from namonexus_fusion.core.temporal_golden_fusion import TemporalGoldenFusion


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def benchmark_data():
    """Generate benchmark dataset once for the entire module."""
    gen = SyntheticSessionGenerator(seed=42)
    profiles, sessions = gen.generate_standard_benchmark()
    return profiles, sessions


@pytest.fixture()
def default_temporal_fusion():
    return TemporalGoldenFusion()


# ===========================================================================
# 1. TemporalConfig
# ===========================================================================


class TestTemporalConfig:
    def test_default_lambda_is_golden_ratio_reciprocal(self):
        cfg = TemporalConfig()
        assert math.isclose(cfg.decay_factor, GOLDEN_RATIO_RECIPROCAL, rel_tol=1e-9)

    def test_lambda_1_means_no_forgetting(self):
        cfg = TemporalConfig(decay_factor=1.0)
        assert cfg.decay_factor == 1.0

    def test_invalid_lambda_raises(self):
        with pytest.raises(Exception):
            TemporalConfig(decay_factor=0.0)

    def test_invalid_lambda_above_1_raises(self):
        with pytest.raises(Exception):
            TemporalConfig(decay_factor=1.1)

    def test_factory_golden(self):
        cfg = TemporalConfig.golden()
        assert math.isclose(cfg.decay_factor, GOLDEN_RATIO_RECIPROCAL)

    def test_factory_adaptive(self):
        cfg = TemporalConfig.adaptive(sensitivity=3.0)
        assert cfg.adaptive_decay is True
        assert cfg.adaptive_sensitivity == 3.0

    def test_factory_realtime(self):
        cfg = TemporalConfig.realtime(decay_rate=0.05)
        assert cfg.time_based_decay is True
        assert cfg.time_decay_rate == 0.05


# ===========================================================================
# 2. TemporalBayesianFilter
# ===========================================================================


class TestTemporalBayesianFilter:
    def test_prior_preserved_with_decay(self):
        """α₀ and β₀ must never change under any decay schedule."""
        alpha0, beta0 = GOLDEN_RATIO * 2.0, 2.0
        f = TemporalBayesianFilter(alpha0, beta0)
        for _ in range(20):
            f.apply_decay(0.5, 0.5, current_score=0.618)
        # Prior parameters unchanged
        assert f.alpha >= alpha0
        assert f.beta >= beta0

    def test_no_forgetting_lambda_1(self):
        """λ = 1 → same as standard Bayesian update (no decay)."""
        alpha0, beta0 = GOLDEN_RATIO * 2.0, 2.0
        cfg = TemporalConfig(decay_factor=1.0)
        f = TemporalBayesianFilter(alpha0, beta0, cfg)

        # Manually compute expected result
        succ, fail = 0.8 * 50, 0.2 * 50
        f.apply_decay(succ, fail, current_score=0.618)
        assert math.isclose(f.alpha, alpha0 + succ, abs_tol=1e-9)
        assert math.isclose(f.beta, beta0 + fail, abs_tol=1e-9)

    def test_forgetting_reduces_old_evidence(self):
        """After many updates with λ < 1, old evidence is discounted."""
        alpha0, beta0 = GOLDEN_RATIO * 2.0, 2.0
        cfg = TemporalConfig(decay_factor=0.5)
        f = TemporalBayesianFilter(alpha0, beta0, cfg)

        # Push many high-score updates
        for _ in range(30):
            f.apply_decay(40.0, 10.0, current_score=0.618)

        # Now push many low-score updates
        for _ in range(10):
            f.apply_decay(5.0, 45.0, current_score=f.alpha / (f.alpha + f.beta))

        # Score should now be closer to low
        score = f.alpha / (f.alpha + f.beta)
        assert score < 0.5

    def test_score_trajectory_records_correctly(self):
        alpha0, beta0 = GOLDEN_RATIO * 2.0, 2.0
        f = TemporalBayesianFilter(alpha0, beta0)
        for _ in range(5):
            f.apply_decay(0.5, 0.5, current_score=0.618)
        traj = f.score_trajectory()
        assert len(traj) == 5
        assert all(isinstance(t, float) and isinstance(s, float) for t, s in traj)

    def test_reset_clears_state(self):
        alpha0, beta0 = GOLDEN_RATIO * 2.0, 2.0
        f = TemporalBayesianFilter(alpha0, beta0)
        for _ in range(10):
            f.apply_decay(0.7, 0.3, current_score=0.618)
        f.reset()
        assert math.isclose(f.alpha, alpha0)
        assert math.isclose(f.beta, beta0)
        assert f.score_trajectory() == []

    def test_effective_obs_decreases_with_small_lambda(self):
        """With λ < 1, effective obs count should stay bounded."""
        alpha0, beta0 = GOLDEN_RATIO * 2.0, 2.0
        cfg = TemporalConfig(decay_factor=0.3)
        f = TemporalBayesianFilter(alpha0, beta0, cfg)
        for _ in range(100):
            f.apply_decay(5.0, 5.0, current_score=0.618)
        # Effective obs should saturate, not grow unboundedly
        assert f.effective_observation_count() < 50

    def test_adaptive_mode_reduces_lambda_on_rapid_change(self):
        """Adaptive mode should use a smaller λ when score changes fast."""
        alpha0, beta0 = GOLDEN_RATIO * 2.0, 2.0
        cfg = TemporalConfig(
            decay_factor=GOLDEN_RATIO_RECIPROCAL,
            adaptive_decay=True,
            adaptive_sensitivity=5.0,
        )
        f = TemporalBayesianFilter(alpha0, beta0, cfg)

        # Simulate rapid oscillation
        for i in range(20):
            score = 0.9 if i % 2 == 0 else 0.1
            f.apply_decay(score * 50, (1 - score) * 50, current_score=score)

        lambdas = f.lambda_history()
        # At least some lambdas should be below the base value
        assert min(lambdas) < GOLDEN_RATIO_RECIPROCAL


# ===========================================================================
# 3. SyntheticSessionGenerator
# ===========================================================================


class TestSyntheticSessionGenerator:
    def test_generates_correct_count(self):
        gen = SyntheticSessionGenerator(seed=0)
        profiles = [SubjectProfile("p1"), SubjectProfile("p2")]
        sessions = gen.generate(profiles, sessions_per_subject=10)
        assert len(sessions) == 20

    def test_all_scores_in_unit_interval(self):
        gen = SyntheticSessionGenerator(seed=1)
        profiles, sessions = gen.generate_standard_benchmark()
        for sess in sessions:
            for obs in sess.observations:
                assert 0.0 <= obs.score <= 1.0
                assert 0.0 <= obs.confidence <= 1.0

    def test_subject_ids_match_profiles(self):
        gen = SyntheticSessionGenerator(seed=2)
        profiles = [SubjectProfile("alpha"), SubjectProfile("beta")]
        sessions = gen.generate(profiles, sessions_per_subject=5)
        subject_ids = {s.subject_id for s in sessions}
        assert subject_ids == {"alpha", "beta"}

    def test_ground_truth_label_present(self):
        gen = SyntheticSessionGenerator(seed=3)
        profiles = [SubjectProfile("x")]
        sessions = gen.generate(profiles, sessions_per_subject=20)
        labels = {s.ground_truth for s in sessions}
        assert labels.issubset({"calm", "stressed", "deceptive"})

    def test_benchmark_produces_5_subjects(self):
        gen = SyntheticSessionGenerator(seed=42)
        profiles, sessions = gen.generate_standard_benchmark()
        assert len(profiles) == 5
        subject_ids = {s.subject_id for s in sessions}
        assert len(subject_ids) == 5


# ===========================================================================
# 4. EmpiricalPriorLearner
# ===========================================================================


class TestEmpiricalPriorLearner:
    def test_returns_golden_ratio_prior_when_no_data(self):
        learner = EmpiricalPriorLearner()
        prior = learner.fit("unknown_subject")
        assert math.isclose(prior.phi_ratio, GOLDEN_RATIO, abs_tol=0.01)
        assert prior.n_sessions_used == 0

    def test_returns_golden_prior_below_min_sessions(self):
        gen = SyntheticSessionGenerator(seed=10)
        profiles = [SubjectProfile("p1")]
        sessions = gen.generate(profiles, sessions_per_subject=3)

        cfg = PriorLearningConfig(min_sessions=5)
        learner = EmpiricalPriorLearner(config=cfg)
        learner.add_sessions(sessions)
        prior = learner.fit("p1")
        assert prior.n_sessions_used == 0

    def test_learned_prior_differs_from_default_for_calm_subject(self, benchmark_data):
        profiles, sessions = benchmark_data
        learner = EmpiricalPriorLearner()
        learner.add_sessions(sessions)
        prior = learner.fit("subject_calm")

        # Calm subject has baseline 0.82 → learned prior_mean should be > 0.618
        assert prior.prior_mean > 0.65, (
            f"Expected prior_mean > 0.65 for calm subject, got {prior.prior_mean:.4f}"
        )

    def test_learned_prior_differs_for_tense_subject(self, benchmark_data):
        profiles, sessions = benchmark_data
        learner = EmpiricalPriorLearner()
        learner.add_sessions(sessions)
        prior = learner.fit("subject_tense")

        # Tense subject has baseline 0.28 → learned prior_mean should be < 0.618
        assert prior.prior_mean < 0.55, (
            f"Expected prior_mean < 0.55 for tense subject, got {prior.prior_mean:.4f}"
        )

    def test_phi_regularization_keeps_ratio_near_phi(self, benchmark_data):
        """High regularization → ratio stays close to φ."""
        profiles, sessions = benchmark_data
        cfg = PriorLearningConfig(phi_regularization=10.0)
        learner = EmpiricalPriorLearner(config=cfg)
        learner.add_sessions(sessions)
        prior = learner.fit("subject_calm")
        assert abs(prior.phi_ratio - GOLDEN_RATIO) < 0.5

    def test_low_regularization_allows_ratio_to_deviate(self, benchmark_data):
        """Low regularization → ratio can deviate from φ to fit data."""
        profiles, sessions = benchmark_data
        cfg = PriorLearningConfig(phi_regularization=0.0)
        learner = EmpiricalPriorLearner(config=cfg)
        learner.add_sessions(sessions)
        prior_calm  = learner.fit("subject_calm")
        prior_tense = learner.fit("subject_tense")
        # The two priors should be meaningfully different
        assert abs(prior_calm.prior_mean - prior_tense.prior_mean) > 0.1

    def test_fit_all_returns_dict(self, benchmark_data):
        profiles, sessions = benchmark_data
        learner = EmpiricalPriorLearner()
        learner.add_sessions(sessions)
        result = learner.fit_all()
        assert len(result) == len(profiles)
        assert all(isinstance(v.alpha0, float) for v in result.values())


# ===========================================================================
# 5. TemporalGoldenFusion (integration)
# ===========================================================================


class TestTemporalGoldenFusion:
    def test_is_backward_compatible(self):
        """All base-class methods should work unchanged."""
        engine = TemporalGoldenFusion()
        engine.update(0.85, 0.70, "text")
        engine.update(0.25, 0.90, "voice")
        assert 0.0 <= engine.fused_score <= 1.0
        lo, hi = engine.credible_interval(0.95)
        assert lo < engine.fused_score < hi
        assert engine.risk_level in ("low", "moderate", "high")
        assert 0.0 <= engine.deception_probability(0.85, 0.70) <= 1.0

    def test_method_chaining(self):
        engine = TemporalGoldenFusion()
        result = engine.update(0.5, 0.5).update(0.7, 0.8)
        assert result is engine

    def test_default_lambda_is_golden_ratio_reciprocal(self):
        engine = TemporalGoldenFusion()
        assert math.isclose(
            engine.temporal_config.decay_factor,
            GOLDEN_RATIO_RECIPROCAL,
            rel_tol=1e-9,
        )

    def test_rapid_state_change_tracked_correctly(self):
        """Engine should respond faster to new evidence than base engine."""
        from namonexus_fusion import GoldenBayesianFusion

        base = GoldenBayesianFusion()
        temporal = TemporalGoldenFusion(
            temporal_config=TemporalConfig(decay_factor=0.5)
        )

        # Warm up with calm observations
        for _ in range(20):
            base.update(0.85, 0.9)
            temporal.update(0.85, 0.9)

        score_base_before    = base.fused_score
        score_temporal_before = temporal.fused_score

        # Now switch to stressed observations
        for _ in range(5):
            base.update(0.15, 0.9)
            temporal.update(0.15, 0.9)

        # Temporal engine should move more than base engine
        drop_base    = score_base_before    - base.fused_score
        drop_temporal = score_temporal_before - temporal.fused_score
        assert drop_temporal > drop_base, (
            f"Temporal drop {drop_temporal:.4f} should exceed base drop {drop_base:.4f}"
        )

    def test_from_learned_prior_uses_personalized_alpha_beta(self, benchmark_data):
        profiles, sessions = benchmark_data
        learner = EmpiricalPriorLearner()
        learner.add_sessions(sessions)
        prior = learner.fit("subject_calm")

        engine = TemporalGoldenFusion.from_learned_prior(prior)
        assert math.isclose(engine.alpha0, prior.alpha0, abs_tol=1e-6)
        assert math.isclose(engine.beta0,  prior.beta0,  abs_tol=1e-6)
        assert engine.learned_prior is prior

    def test_personalized_prior_gives_better_cold_start_for_calm(self, benchmark_data):
        """For a calm subject, personalized prior_mean > default prior_mean."""
        profiles, sessions = benchmark_data
        learner = EmpiricalPriorLearner()
        learner.add_sessions(sessions)
        prior = learner.fit("subject_calm")

        default_engine = TemporalGoldenFusion()
        personal_engine = TemporalGoldenFusion.from_learned_prior(prior)

        # Before any observations, personalized engine should start higher
        assert personal_engine.fused_score > default_engine.fused_score

    def test_reset_restores_learned_prior(self, benchmark_data):
        profiles, sessions = benchmark_data
        learner = EmpiricalPriorLearner()
        learner.add_sessions(sessions)
        prior = learner.fit("subject_calm")

        engine = TemporalGoldenFusion.from_learned_prior(prior)
        initial_score = engine.fused_score

        engine.update(0.1, 1.0)
        engine.reset()

        assert math.isclose(engine.fused_score, initial_score, abs_tol=1e-9)

    def test_lambda_history_populated(self):
        engine = TemporalGoldenFusion()
        for _ in range(5):
            engine.update(0.5, 0.5)
        history = engine.lambda_history()
        assert len(history) == 5
        assert all(0.0 < lam <= 1.0 for lam in history)

    def test_effective_obs_count_increases(self):
        engine = TemporalGoldenFusion()
        before = engine.effective_observation_count
        engine.update(0.5, 1.0)
        after = engine.effective_observation_count
        assert after > before

    def test_state_serialisation_includes_temporal(self):
        engine = TemporalGoldenFusion()
        engine.update(0.6, 0.7)
        state = engine.get_state()
        assert "temporal" in state.config
        assert "decay_factor" in state.config["temporal"]

    def test_adaptive_temporal_config(self):
        engine = TemporalGoldenFusion(
            temporal_config=TemporalConfig.adaptive(sensitivity=3.0)
        )
        for _ in range(10):
            engine.update(0.5, 0.5)
        assert 0.0 <= engine.fused_score <= 1.0

    @given(
        score=st.floats(min_value=0.0, max_value=1.0),
        confidence=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_property_score_always_in_unit_interval(self, score, confidence):
        engine = TemporalGoldenFusion()
        engine.update(score, confidence)
        assert 0.0 <= engine.fused_score <= 1.0

    @given(
        score=st.floats(min_value=0.0, max_value=1.0),
        confidence=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_property_uncertainty_non_negative(self, score, confidence):
        engine = TemporalGoldenFusion()
        engine.update(score, confidence)
        assert engine.uncertainty >= 0.0
