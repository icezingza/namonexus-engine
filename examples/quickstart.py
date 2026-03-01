"""
NamoNexus Fusion Engine — Quick-Start Example
==============================================
Patent-Pending Technology | NamoNexus Research Team

Run this script to see all four phases in action:

    python examples/quickstart.py
"""

import json
import math

# ─── Phase constants ────────────────────────────────────────────────────────
from namonexus_fusion import NamoNexusEngine
from namonexus_fusion.core import (
    GOLDEN_RATIO,
    HierarchicalBayesianModel,
    EmpiricalPriorLearner,
    SyntheticSessionGenerator,
    SubjectProfile,
    StreamingObservation,
    InMemoryConnector,
)

PHI = GOLDEN_RATIO
DIVIDER = "─" * 60


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ────────────────────────────────────────────────────────────────────────────
# 1.  Base fusion (Claims 1–5)
# ────────────────────────────────────────────────────────────────────────────
section("Phase v3.0 — Base Golden Bayesian Fusion")

engine = NamoNexusEngine()
print(f"Golden Ratio prior: α₀/β₀ = {engine._alpha0:.4f}/{engine._beta0:.4f}"
      f" = {engine._alpha0/engine._beta0:.6f} (φ = {PHI:.6f})")

engine.update_batch([
    {"score": 0.85, "confidence": 0.70, "modality": "text"},
    {"score": 0.25, "confidence": 0.90, "modality": "voice"},
    {"score": 0.60, "confidence": 0.85, "modality": "face"},
])

lo, hi = engine.credible_interval()
print(f"Fused score  : {engine.fused_score:.4f}")
print(f"Risk level   : {engine.risk_level()}")
print(f"Uncertainty  : ±{engine.uncertainty:.4f}")
print(f"95% CI       : [{lo:.4f}, {hi:.4f}]")
print(f"Deception P  : {engine.deception_probability():.4f}")


# ────────────────────────────────────────────────────────────────────────────
# 2.  Personalised prior (Claim 7)
# ────────────────────────────────────────────────────────────────────────────
section("Phase 1 — Personalised Prior via φ-Regularised MLE")

gen = SyntheticSessionGenerator(seed=42)
profiles = [SubjectProfile("user_calm", baseline_score=0.80)]
sessions = gen.generate(profiles, sessions_per_subject=20)

learner = EmpiricalPriorLearner()
learner.add_sessions(sessions)
prior = learner.fit("user_calm")

print(f"Learned prior: α₀={prior.alpha0:.4f}, β₀={prior.beta0:.4f}")
print(f"Prior mean   : {prior.prior_mean:.4f}  (universal default: {PHI/(1+PHI):.4f})")
print(f"φ deviation  : {prior.golden_ratio_deviation:.4f}")

eng1 = NamoNexusEngine(learned_prior=prior)
eng1.update(0.80, 0.70, "text")
print(f"Fused score with personalised prior: {eng1.fused_score:.4f}")


# ────────────────────────────────────────────────────────────────────────────
# 3.  Drift detection + streaming (Claims 11–12)
# ────────────────────────────────────────────────────────────────────────────
section("Phase 3 — Drift Detection + Streaming Inference")

eng3 = NamoNexusEngine()
pipeline = eng3.stream()

observations = [
    StreamingObservation(score=0.80, confidence=0.90, modality="text"),
    StreamingObservation(score=0.30, confidence=0.75, modality="voice"),
    StreamingObservation(score=0.65, confidence=0.80, modality="face"),
    StreamingObservation(score=0.82, confidence=0.88, modality="text"),
    StreamingObservation(score=0.28, confidence=0.70, modality="voice"),
]

results = pipeline.run_sync(InMemoryConnector(observations))
print(f"Processed {len(results)} streaming observations")
for r in results:
    trend = r.window_stats.risk_trend if r.window_stats else "n/a"
    print(f"  score={r.fused_score:.4f}  risk={r.risk_level:<8}  trend={trend}")

if eng3.has_drift_alarm:
    print("⚠ Drift alarm triggered")
else:
    print("No drift alarm (normal operating range)")


# ────────────────────────────────────────────────────────────────────────────
# 4.  XAI Explanation (Claim 13)
# ────────────────────────────────────────────────────────────────────────────
section("Phase 4 — Shapley XAI + GDPR/PDPA Compliance Report")

eng4 = NamoNexusEngine()
eng4.update(0.85, 0.70, "text")
eng4.update(0.25, 0.90, "voice")
eng4.update(0.60, 0.85, "face")

report = eng4.explain(metadata={"session_id": "demo_001", "operator": "quickstart.py"})
print(f"\nCompliance Summary:\n  {report.summary_text}")
print(f"\nDominant modality: {report.dominant_modality}")
print(f"GDPR compliant  : {report.compliance_gdpr}")
print(f"PDPA compliant  : {report.compliance_pdpa}")
print("\nPer-modality Shapley attributions:")
for attr in report.modality_attributions:
    print(f"  {attr.modality:<8}  φ={attr.shapley_value:+.4f}  "
          f"alignment={attr.alignment.value:<15}  {attr.explanation}")

audit = report.to_dict()
print(f"\nAudit record keys: {list(audit.keys())}")


# ────────────────────────────────────────────────────────────────────────────
# 5.  Hierarchical Bayesian + Federated Learning (Claim 14)
# ────────────────────────────────────────────────────────────────────────────
section("Phase 4 — Hierarchical Bayesian + Federated Learning")

# Site A
model_a = HierarchicalBayesianModel(site_id="site_a")
eng_a   = NamoNexusEngine(hierarchical_model=model_a, subject_id="user_001")
eng_a.update(0.8, 0.9, "text")
eng_a.update(0.4, 0.7, "voice")
eng_a.commit_session()
delta_a = model_a.export_federated_delta()
print(f"Site A delta: Δα={delta_a.alpha_delta:.4f}  Δβ={delta_a.beta_delta:.4f}")

# Site B
model_b = HierarchicalBayesianModel(site_id="site_b")
eng_b   = NamoNexusEngine(hierarchical_model=model_b, subject_id="user_002")
eng_b.update(0.7, 0.85, "text")
eng_b.update(0.5, 0.75, "face")
eng_b.commit_session()
delta_b = model_b.export_federated_delta()
print(f"Site B delta: Δα={delta_b.alpha_delta:.4f}  Δβ={delta_b.beta_delta:.4f}")

# Coordinator
coordinator = HierarchicalBayesianModel(site_id="coordinator")
coordinator.aggregate_federated_deltas([delta_a, delta_b])
pop = coordinator._population_prior
print(f"\nCoordinator population prior after aggregation:")
print(f"  α_pop/β_pop = {pop.alpha:.4f}/{pop.beta:.4f} = {pop.alpha/pop.beta:.6f}")
print(f"  φ constraint: {abs(pop.alpha/pop.beta - PHI):.2e} from φ={PHI:.6f}")

# New user warm-starts from enriched population prior
eng_new = NamoNexusEngine(hierarchical_model=coordinator, subject_id="new_user")
blended = eng_new.blended_prior()
print(f"\nNew user blended prior: ρ={blended.rho:.4f}  "
      f"mean={blended.alpha_blend/(blended.alpha_blend+blended.beta_blend):.4f}")


# ────────────────────────────────────────────────────────────────────────────
# 6.  Session Summary
# ────────────────────────────────────────────────────────────────────────────
section("Full Session Summary")

summary = eng4.session_summary()
print(json.dumps(
    {k: v for k, v in summary.items() if k not in ("calibration", "trust", "drift")},
    indent=2,
))

print(f"\n{'─'*60}")
print("  ✅ All phases demonstrated successfully.")
print(f"{'─'*60}\n")
