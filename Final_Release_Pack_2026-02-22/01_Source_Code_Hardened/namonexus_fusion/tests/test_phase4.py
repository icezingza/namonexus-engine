"""
Phase 4 Tests — Explainability Layer (XAI) + Hierarchical Bayesian Model
=========================================================================
Patent-Pending Technology | NamoNexus Research Team

Run:
    pytest namonexus_fusion/tests/test_phase4.py -v --tb=short

Tests cover:
  - XAIConfig (golden defaults, Golden Ratio thresholds)
  - ExplainabilityLayer (record, attribute, report structure)
  - Shapley attribution invariants (signs, normalization, dominance)
  - Compliance text (GDPR/PDPA fields present)
  - HierarchicalConfig (validation, factories)
  - PopulationPrior (phi constraint, strength)
  - IndividualPosterior (warm start, accumulated evidence)
  - HierarchicalBayesianModel (register, update, blend, federated)
  - BlendedPrior (rho transitions, cold/warm/individual)
  - FederatedDelta (export, round-trip serialization)
  - Federated aggregation (multi-site, phi constraint enforcement)
  - Phase4GoldenFusion (integrated engine, XAI + hierarchical)
  - Full compliance audit trail
"""

from __future__ import annotations

import json
import math
import time
import pytest

from namonexus_fusion.core.explainability import (
    ExplainabilityLayer,
    ExplanationReport,
    XAIConfig,
    ModalityAttribution,
    ModalityAlignment,
    PriorAttribution,
    _PHI,
    _PHI_RECIP,
)

from namonexus_fusion.core.hierarchical_bayesian import (
    HierarchicalBayesianModel,
    HierarchicalConfig,
    PopulationPrior,
    IndividualPosterior,
    BlendedPrior,
    FederatedDelta,
    _PHI     as HIER_PHI,
    _PHI_SQ  as HIER_PHI_SQ,
    _PHI_RECIP as HIER_PHI_RECIP,
)

from namonexus_fusion.core.phase4_fusion import Phase4GoldenFusion


# ===========================================================================
# Helpers
# ===========================================================================

def make_engine(**kwargs) -> Phase4GoldenFusion:
    return Phase4GoldenFusion(**kwargs)


def feed_engine(engine: Phase4GoldenFusion, n: int = 3) -> None:
    engine.update(0.85, 0.80, "text")
    if n >= 2:
        engine.update(0.30, 0.90, "voice")
    if n >= 3:
        engine.update(0.65, 0.85, "face")


# ===========================================================================
# 1.  XAIConfig
# ===========================================================================


class TestXAIConfig:
    def test_consistent_threshold_uses_phi_sq_recip(self):
        cfg      = XAIConfig.default()
        expected = _PHI_RECIP ** 2   # 1/phi^2 ≈ 0.382
        assert math.isclose(cfg.consistent_threshold, expected, rel_tol=1e-6)

    def test_contradicting_threshold_uses_phi_recip(self):
        cfg = XAIConfig.default()
        assert math.isclose(cfg.contradicting_threshold, _PHI_RECIP, rel_tol=1e-6)

    def test_consistent_less_than_contradicting(self):
        cfg = XAIConfig.default()
        assert cfg.consistent_threshold < cfg.contradicting_threshold

    def test_to_dict_includes_thresholds(self):
        d = XAIConfig.default().to_dict()
        assert "consistent_threshold"    in d
        assert "contradicting_threshold" in d


# ===========================================================================
# 2.  ExplainabilityLayer via Phase4GoldenFusion
# ===========================================================================


class TestExplainabilityLayer:
    def test_explain_returns_report(self):
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain()
        assert isinstance(report, ExplanationReport)

    def test_report_contains_all_fed_modalities(self):
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain()
        mods   = {a.modality for a in report.modality_attributions}
        assert "text"  in mods
        assert "voice" in mods
        assert "face"  in mods

    def test_fused_score_in_unit_interval(self):
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain()
        assert 0.0 <= report.fused_score <= 1.0

    def test_normalized_attributions_bounded(self):
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain()
        for attr in report.modality_attributions:
            assert -1.0 <= attr.normalized_attribution <= 1.0

    def test_positive_contributions_sum_within_100(self):
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain()
        total  = sum(a.contribution_pct for a in report.modality_attributions)
        assert 0.0 <= total <= 100.0 + 1e-6

    def test_risk_level_valid_value(self):
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain()
        assert report.risk_level in ("low", "medium", "high")

    def test_prior_attribution_present(self):
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain()
        assert isinstance(report.prior_attribution, PriorAttribution)

    def test_prior_phi_ratio_near_phi(self):
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain()
        assert math.isclose(
            report.prior_attribution.phi_ratio, _PHI, abs_tol=0.5
        )

    def test_summary_text_contains_risk_level(self):
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain()
        assert report.risk_level.upper() in report.summary_text

    def test_explanation_sentence_per_modality(self):
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain()
        for attr in report.modality_attributions:
            assert len(attr.explanation_sentence) > 20
            assert attr.modality in attr.explanation_sentence

    def test_timestamp_is_iso8601(self):
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain()
        assert "T" in report.timestamp

    def test_empty_engine_returns_valid_report(self):
        engine = make_engine()
        report = engine.explain()
        assert isinstance(report, ExplanationReport)
        assert report.modality_attributions == []

    def test_window_limits_observation_scope(self):
        engine = make_engine()
        for _ in range(10):
            engine.update(0.7, 0.8, "text")
        for _ in range(5):
            engine.update(0.3, 0.8, "voice")
        # Window=5 covers only voice observations
        report = engine.explain(window=5)
        mods   = {a.modality for a in report.modality_attributions}
        assert "voice" in mods

    def test_metadata_attached_to_report(self):
        engine = make_engine()
        feed_engine(engine)
        meta   = {"session_id": "s001", "operator": "audit_bot"}
        report = engine.explain(metadata=meta)
        assert report.metadata["session_id"] == "s001"
        assert report.metadata["operator"]   == "audit_bot"

    def test_report_to_dict_json_serializable(self):
        engine = make_engine()
        feed_engine(engine)
        report   = engine.explain()
        d        = report.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 100

    def test_reset_clears_xai_records(self):
        engine = make_engine()
        feed_engine(engine)
        assert len(engine.xai._records) > 0
        engine.reset()
        assert len(engine.xai._records) == 0

    def test_report_history_accumulates(self):
        engine = make_engine()
        feed_engine(engine)
        engine.explain()
        engine.explain()
        assert len(engine.xai.report_history()) == 2

    def test_alignment_contradicting_on_extreme_divergence(self):
        engine = make_engine()
        for _ in range(5):
            engine.update(0.85, 0.90, "text")
        engine.update(0.05, 0.90, "voice")   # extreme drop
        report = engine.explain(window=1)
        voice_attr = next(
            (a for a in report.modality_attributions if a.modality == "voice"), None
        )
        assert voice_attr is not None
        assert voice_attr.alignment in (
            ModalityAlignment.MIXED, ModalityAlignment.CONTRADICTING
        )


# ===========================================================================
# 3.  Shapley attribution invariants
# ===========================================================================


class TestShapleyInvariants:
    def test_score_above_aggregate_gives_positive_marginal(self):
        engine = make_engine()
        for _ in range(5):
            engine.update(0.20, 0.90, "voice")
        engine.update(0.95, 0.90, "text")
        report = engine.explain(window=1)
        text_attr = next(
            (a for a in report.modality_attributions if a.modality == "text"), None
        )
        assert text_attr is not None
        assert text_attr.marginal_contribution >= 0.0

    def test_score_below_aggregate_gives_negative_marginal(self):
        engine = make_engine()
        for _ in range(5):
            engine.update(0.85, 0.90, "text")
        engine.update(0.05, 0.90, "voice")
        report = engine.explain(window=1)
        voice_attr = next(
            (a for a in report.modality_attributions if a.modality == "voice"), None
        )
        assert voice_attr is not None
        assert voice_attr.marginal_contribution <= 0.0

    def test_score_equal_aggregate_gives_consistent_alignment(self):
        engine = make_engine()
        for _ in range(10):
            engine.update(0.70, 0.80, "text")
        fused = engine.fused_score
        engine.update(fused, 0.80, "face")
        report = engine.explain(window=1)
        face_attr = next(
            (a for a in report.modality_attributions if a.modality == "face"), None
        )
        if face_attr:
            assert face_attr.alignment == ModalityAlignment.CONSISTENT


# ===========================================================================
# 4.  HierarchicalConfig
# ===========================================================================


class TestHierarchicalConfig:
    def test_golden_factory_uses_phi_sq(self):
        cfg = HierarchicalConfig.golden()
        assert math.isclose(cfg.population_prior_strength, HIER_PHI_SQ, rel_tol=1e-6)

    def test_federated_factory_sets_dp_noise(self):
        cfg = HierarchicalConfig.federated(dp_noise=0.2)
        assert cfg.dp_noise_scale == 0.2

    def test_invalid_strength_raises(self):
        with pytest.raises((ValueError, Exception)):
            HierarchicalConfig(population_prior_strength=0.0)

    def test_invalid_tau_raises(self):
        with pytest.raises((ValueError, Exception)):
            HierarchicalConfig(tau=-1.0)

    def test_to_dict_contains_tau(self):
        d = HierarchicalConfig.golden().to_dict()
        assert "tau" in d


# ===========================================================================
# 5.  Population prior
# ===========================================================================


class TestPopulationPrior:
    def test_initial_phi_ratio_is_phi(self):
        model = HierarchicalBayesianModel()
        pop   = model.population_prior
        assert math.isclose(pop.phi_ratio, HIER_PHI, abs_tol=0.01)

    def test_population_mean_near_phi_recip(self):
        # With phi ratio: mean = phi/(1+phi) ≈ 0.618
        model    = HierarchicalBayesianModel()
        expected = HIER_PHI / (1.0 + HIER_PHI)
        assert math.isclose(model.population_prior.mean, expected, abs_tol=0.01)

    def test_initial_strength_scales_with_config(self):
        s     = 3.0
        model = HierarchicalBayesianModel(
            config=HierarchicalConfig(population_prior_strength=s)
        )
        pop   = model.population_prior
        # alpha_pop = phi * s, beta_pop = s  → strength = (phi+1) * s = phi^2 * s
        assert math.isclose(pop.strength, HIER_PHI_SQ * s, rel_tol=0.01)


# ===========================================================================
# 6.  Individual management
# ===========================================================================


class TestIndividualManagement:
    def test_register_creates_entry(self):
        model = HierarchicalBayesianModel()
        model.register_subject("u1")
        assert "u1" in model.all_subjects()

    def test_register_idempotent(self):
        model = HierarchicalBayesianModel()
        p1    = model.register_subject("u1")
        p2    = model.register_subject("u1")
        assert p1.subject_id == p2.subject_id

    def test_warm_start_matches_population_prior(self):
        model = HierarchicalBayesianModel()
        pop   = model.population_prior
        ind   = model.register_subject("u1")
        assert math.isclose(ind.alpha_ind, pop.alpha_pop, rel_tol=1e-6)
        assert math.isclose(ind.beta_ind,  pop.beta_pop,  rel_tol=1e-6)

    def test_update_accumulates_evidence(self):
        model = HierarchicalBayesianModel()
        model.register_subject("u1")
        ind   = model.update_individual("u1", alpha_delta=2.0, beta_delta=0.5, n_obs=5)
        assert ind.accumulated_evidence > 0
        assert ind.n_observations == 5

    def test_update_auto_registers(self):
        model = HierarchicalBayesianModel()
        model.update_individual("new_user", alpha_delta=1.0, beta_delta=1.0, n_obs=2)
        assert "new_user" in model.all_subjects()

    def test_max_strength_cap_applied(self):
        model = HierarchicalBayesianModel(
            config=HierarchicalConfig(max_individual_strength=10.0)
        )
        model.register_subject("u1")
        for _ in range(100):
            model.update_individual("u1", alpha_delta=2.0, beta_delta=1.0, n_obs=1)
        ind = model.individual_posterior("u1")
        assert ind.alpha_ind + ind.beta_ind <= 10.0 + 1e-6


# ===========================================================================
# 7.  Blended prior (rho transitions)
# ===========================================================================


class TestBlendedPrior:
    def test_new_subject_gets_population_prior(self):
        model   = HierarchicalBayesianModel()
        pop     = model.population_prior
        blended = model.get_blended_prior("brand_new")
        assert math.isclose(blended.alpha_blend, pop.alpha_pop, rel_tol=1e-6)
        assert blended.rho    == 0.0
        assert blended.source == "population"

    def test_well_observed_subject_trends_individual(self):
        model = HierarchicalBayesianModel(config=HierarchicalConfig(tau=1.0))
        model.register_subject("veteran")
        for _ in range(50):
            model.update_individual("veteran", alpha_delta=1.0, beta_delta=0.5,
                                    n_obs=5, n_sessions=1)
        blended = model.get_blended_prior("veteran")
        assert blended.rho > 0.5

    def test_rho_monotone_with_observations(self):
        model = HierarchicalBayesianModel(config=HierarchicalConfig(tau=2.0))
        model.register_subject("u")
        rho_before = model.get_blended_prior("u").rho
        model.update_individual("u", alpha_delta=1.0, beta_delta=0.5,
                                n_obs=20, n_sessions=5)
        rho_after  = model.get_blended_prior("u").rho
        assert rho_after >= rho_before

    def test_rho_bounded_to_unit_interval(self):
        model = HierarchicalBayesianModel()
        model.register_subject("u")
        model.update_individual("u", alpha_delta=100.0, beta_delta=50.0,
                                n_obs=500, n_sessions=100)
        blended = model.get_blended_prior("u")
        assert 0.0 <= blended.rho <= 1.0

    def test_min_sessions_gate_respected(self):
        model = HierarchicalBayesianModel(
            config=HierarchicalConfig(min_sessions_for_blend=5, tau=1.0)
        )
        model.register_subject("u")
        model.update_individual("u", alpha_delta=5.0, beta_delta=1.0,
                                n_obs=50, n_sessions=2)   # < 5 sessions
        blended = model.get_blended_prior("u")
        assert blended.rho == 0.0

    def test_phi_sq_tau_gives_rho_half(self):
        """At n = phi^2 * tau observations, rho should equal 0.5 exactly."""
        tau   = 5.0
        model = HierarchicalBayesianModel(
            config=HierarchicalConfig(tau=tau, min_sessions_for_blend=1)
        )
        model.register_subject("u")
        n_half = int(HIER_PHI_SQ * tau)
        model.update_individual("u", alpha_delta=1.0, beta_delta=0.5,
                                n_obs=n_half, n_sessions=10)
        blended = model.get_blended_prior("u")
        assert math.isclose(blended.rho, 0.5, abs_tol=0.05)


# ===========================================================================
# 8.  Federated aggregation
# ===========================================================================


class TestFederatedAggregation:
    def test_export_delta_structure(self):
        model = HierarchicalBayesianModel()
        model.update_individual("u1", alpha_delta=2.0, beta_delta=1.0, n_obs=5, n_sessions=1)
        delta = model.export_federated_delta()
        assert isinstance(delta, FederatedDelta)
        assert delta.n_subjects >= 1
        assert delta.alpha_delta >= 0.0

    def test_delta_serialization_roundtrip(self):
        model    = HierarchicalBayesianModel()
        model.update_individual("u1", alpha_delta=1.5, beta_delta=0.8, n_obs=3, n_sessions=1)
        delta    = model.export_federated_delta()
        restored = FederatedDelta.from_json(delta.to_json())
        assert math.isclose(restored.alpha_delta, delta.alpha_delta, rel_tol=1e-6)
        assert restored.site_id == delta.site_id

    def test_aggregate_updates_population(self):
        model_a = HierarchicalBayesianModel(site_id="site_A")
        model_b = HierarchicalBayesianModel(site_id="site_B")

        # Both sites accumulate observations
        model_a.update_individual("u1", alpha_delta=3.0, beta_delta=1.0, n_obs=10, n_sessions=2)
        model_b.update_individual("u2", alpha_delta=2.0, beta_delta=1.5, n_obs=8,  n_sessions=2)

        delta_a = model_a.export_federated_delta()
        delta_b = model_b.export_federated_delta()

        # Coordinator aggregates into a third model
        coord = HierarchicalBayesianModel(site_id="coordinator")
        pop_before = coord.population_prior.alpha_pop
        coord.aggregate_federated_deltas([delta_a, delta_b])
        pop_after  = coord.population_prior.alpha_pop

        # Population prior should have changed
        assert pop_after != pop_before

    def test_phi_constraint_enforced_after_aggregation(self):
        coord = HierarchicalBayesianModel(
            config=HierarchicalConfig(enforce_phi_constraint=True, phi_tolerance=0.05)
        )
        # Inject a delta that would violate the phi ratio
        skewed_delta = FederatedDelta(
            site_id     = "test",
            alpha_delta = 50.0,  # Very high — would push ratio away from phi
            beta_delta  = 0.1,
            n_subjects  = 1,
            n_sessions  = 1,
        )
        coord.aggregate_federated_deltas([skewed_delta])
        phi_ratio = coord.population_prior.phi_ratio
        # After enforcement, ratio should be near phi (within a reasonable range)
        assert abs(phi_ratio - HIER_PHI) < 0.5, (
            f"phi_ratio {phi_ratio:.4f} drifted too far from phi {HIER_PHI:.4f}"
        )

    def test_empty_deltas_leaves_population_unchanged(self):
        model      = HierarchicalBayesianModel()
        pop_before = model.population_prior.alpha_pop
        model.aggregate_federated_deltas([])
        pop_after  = model.population_prior.alpha_pop
        assert math.isclose(pop_before, pop_after)

    def test_dp_noise_applied_when_configured(self):
        model = HierarchicalBayesianModel(
            config=HierarchicalConfig(dp_noise_scale=1.0)
        )
        model.update_individual("u1", alpha_delta=2.0, beta_delta=1.0, n_obs=5, n_sessions=1)
        delta = model.export_federated_delta()
        assert delta.noise_applied is True
        assert delta.noise_scale   == 1.0

    def test_version_increments_on_aggregation(self):
        model = HierarchicalBayesianModel()
        v0    = model.population_prior.version
        delta = FederatedDelta(
            site_id="s", alpha_delta=1.0, beta_delta=0.5, n_subjects=1, n_sessions=1
        )
        model.aggregate_federated_deltas([delta])
        assert model.population_prior.version == v0 + 1

    def test_multi_site_aggregation_averages_deltas(self):
        """Aggregating N identical deltas should not be N× worse than one."""
        coord = HierarchicalBayesianModel(
            config=HierarchicalConfig(enforce_phi_constraint=False)
        )
        pop_start = coord.population_prior.alpha_pop
        delta = FederatedDelta(
            site_id="s", alpha_delta=2.0, beta_delta=1.0, n_subjects=1, n_sessions=1
        )
        # Aggregate same delta 3 times (simulating 3 sites with same pattern)
        coord.aggregate_federated_deltas([delta, delta, delta])
        pop_end = coord.population_prior.alpha_pop
        # Population should have increased, but not by 3× the single-delta amount
        single_increase = delta.alpha_delta / delta.n_subjects   # per-site mean
        triple_increase = pop_end - pop_start
        assert triple_increase > 0
        # Triple increase should be close to single increase (averaged over 3 sites)
        assert math.isclose(triple_increase, single_increase, rel_tol=0.01)


# ===========================================================================
# 9.  Phase4GoldenFusion — integrated engine
# ===========================================================================


class TestPhase4Integrated:
    def test_instantiation(self):
        engine = make_engine()
        assert engine is not None

    def test_update_returns_self(self):
        engine = make_engine()
        assert engine.update(0.8, 0.9, "text") is engine

    def test_fused_score_in_unit_interval(self):
        engine = make_engine()
        feed_engine(engine)
        assert 0.0 <= engine.fused_score <= 1.0

    def test_explain_accessible(self):
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain()
        assert isinstance(report, ExplanationReport)

    def test_xai_property_returns_layer(self):
        engine = make_engine()
        assert isinstance(engine.xai, ExplainabilityLayer)

    def test_repr_contains_xai_records(self):
        engine = make_engine()
        feed_engine(engine)
        r = repr(engine)
        assert "xai_records=" in r

    def test_hierarchical_warm_start(self):
        model   = HierarchicalBayesianModel()
        pop     = model.population_prior
        engine  = Phase4GoldenFusion(
            hierarchical_model=model,
            subject_id="user_001",
        )
        # Fused score at init should reflect the population prior mean
        expected_mean = pop.mean
        assert math.isclose(engine.fused_score, expected_mean, abs_tol=0.05)

    def test_commit_session_updates_individual(self):
        model   = HierarchicalBayesianModel()
        engine  = Phase4GoldenFusion(
            hierarchical_model=model,
            subject_id="user_001",
        )
        feed_engine(engine)
        posterior = engine.commit_session()
        assert posterior is not None
        assert posterior.n_sessions >= 1

    def test_commit_session_no_model_returns_none(self):
        engine = make_engine()
        feed_engine(engine)
        result = engine.commit_session()
        assert result is None

    def test_blended_prior_none_without_model(self):
        engine = make_engine()
        assert engine.blended_prior() is None

    def test_blended_prior_with_model(self):
        model  = HierarchicalBayesianModel()
        engine = Phase4GoldenFusion(hierarchical_model=model, subject_id="u1")
        bp     = engine.blended_prior()
        assert isinstance(bp, BlendedPrior)

    def test_subject_id_property(self):
        engine = Phase4GoldenFusion(
            hierarchical_model=HierarchicalBayesianModel(),
            subject_id="sub_42",
        )
        assert engine.subject_id == "sub_42"

    def test_hierarchical_model_property(self):
        model  = HierarchicalBayesianModel()
        engine = Phase4GoldenFusion(hierarchical_model=model, subject_id="u1")
        assert engine.hierarchical_model is model

    def test_multiple_commits_increase_individual_evidence(self):
        model  = HierarchicalBayesianModel()
        engine = Phase4GoldenFusion(hierarchical_model=model, subject_id="u1")

        feed_engine(engine, n=3)
        engine.commit_session()
        ind_after_1 = model.individual_posterior("u1").accumulated_evidence

        feed_engine(engine, n=3)
        engine.commit_session()
        ind_after_2 = model.individual_posterior("u1").accumulated_evidence

        assert ind_after_2 >= ind_after_1

    def test_federated_workflow_end_to_end(self):
        """
        Full federated cycle:
          1. Two sites train on local subjects.
          2. Each exports a FederatedDelta.
          3. Coordinator aggregates deltas.
          4. New subject at coordinator gets warm-started from updated pop prior.
        """
        # Site A
        model_a = HierarchicalBayesianModel(site_id="site_A")
        eng_a   = Phase4GoldenFusion(hierarchical_model=model_a, subject_id="a1")
        for _ in range(5):
            eng_a.update(0.80, 0.85, "text")
        eng_a.commit_session()
        delta_a = model_a.export_federated_delta()

        # Site B
        model_b = HierarchicalBayesianModel(site_id="site_B")
        eng_b   = Phase4GoldenFusion(hierarchical_model=model_b, subject_id="b1")
        for _ in range(5):
            eng_b.update(0.40, 0.85, "voice")
        eng_b.commit_session()
        delta_b = model_b.export_federated_delta()

        # Coordinator
        coord   = HierarchicalBayesianModel(site_id="coordinator")
        coord.aggregate_federated_deltas([delta_a, delta_b])

        # New user at coordinator gets a population-informed warm start
        new_engine = Phase4GoldenFusion(
            hierarchical_model=coord,
            subject_id="new_user_at_coord",
        )
        report = new_engine.explain()
        assert isinstance(report, ExplanationReport)
        # Engine should function normally (no crash, valid score)
        assert 0.0 <= new_engine.fused_score <= 1.0

    def test_compliance_report_fields(self):
        """Key fields required for GDPR Article 22 / PDPA Section 40 audit."""
        engine = make_engine()
        feed_engine(engine)
        report = engine.explain(metadata={"session_id": "audit_001"})
        d      = report.to_dict()

        # Required for compliance
        assert "timestamp"             in d
        assert "fused_score"           in d
        assert "risk_level"            in d
        assert "modality_attributions" in d
        assert "prior_attribution"     in d
        assert "summary_text"          in d
        assert "metadata"              in d
        assert d["metadata"]["session_id"] == "audit_001"

        # Each modality attribution must be individually auditable
        for attr in d["modality_attributions"]:
            assert "modality"               in attr
            assert "raw_score"              in attr
            assert "raw_confidence"         in attr
            assert "normalized_attribution" in attr
            assert "alignment"              in attr
            assert "explanation_sentence"   in attr


# ===========================================================================
# 10.  Golden Ratio structural invariants (Phase 4)
# ===========================================================================


class TestGoldenRatioInvariantsPhase4:
    def test_phi_sq_equals_phi_plus_one(self):
        assert math.isclose(HIER_PHI_SQ, HIER_PHI + 1.0, rel_tol=1e-9)

    def test_phi_recip_plus_phi_sq_recip_equals_one(self):
        assert math.isclose(HIER_PHI_RECIP + (1.0 / HIER_PHI_SQ), 1.0, rel_tol=1e-9)

    def test_population_prior_phi_ratio_is_phi(self):
        model = HierarchicalBayesianModel()
        assert math.isclose(model.population_prior.phi_ratio, HIER_PHI, abs_tol=0.001)

    def test_xai_consistent_threshold_is_phi_sq_recip(self):
        cfg      = XAIConfig.default()
        expected = _PHI_RECIP ** 2
        assert math.isclose(cfg.consistent_threshold, expected, rel_tol=1e-6)

    def test_xai_contradicting_threshold_is_phi_recip(self):
        cfg = XAIConfig.default()
        assert math.isclose(cfg.contradicting_threshold, _PHI_RECIP, rel_tol=1e-6)

    def test_rho_at_phi_sq_tau_equals_half(self):
        """phi^2 * tau is the exact transition point where rho = 0.5."""
        tau   = 4.0
        model = HierarchicalBayesianModel(
            config=HierarchicalConfig(tau=tau, min_sessions_for_blend=1)
        )
        model.register_subject("u")
        n_half = int(HIER_PHI_SQ * tau)
        model.update_individual("u", alpha_delta=1.0, beta_delta=0.5,
                                n_obs=n_half, n_sessions=10)
        rho = model.get_blended_prior("u").rho
        assert math.isclose(rho, 0.5, abs_tol=0.05)
