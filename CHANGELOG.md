# Changelog

All notable changes to the NamoNexus Fusion Engine are documented in this file.

---

## [4.0.0] — Phase 4: Commercialization Layer

### Added — Feature 4.1: Explainability Layer (Patent Claim 13)
- `ExplainabilityLayer` — records every `update()` call for post-hoc attribution
- `ExplanationReport` — JSON-serialisable compliance audit record (GDPR Article 22 / PDPA Section 40)
- Shapley marginal attribution: `φ_m = w_m · (score_m − fused_before)`
- Golden Ratio prior baseline normalization — attributions are relative to the φ prior contribution
- Alignment classification with thresholds at `1/φ²` (CONSISTENT) and `1/φ` (CONTRADICTING)
- Per-modality human-readable explanation sentences
- `Phase4GoldenFusion.explain()` — generates `ExplanationReport`

### Added — Feature 4.2: Hierarchical Bayesian Model (Patent Claim 14)
- `HierarchicalBayesianModel` — two-level Beta hierarchy with federated aggregation
- `PopulationPrior` — population-level Beta prior with φ ratio constraint (`α_pop/β_pop = φ`)
- `IndividualPosterior` — per-subject posterior with warm-start from population prior
- `BlendedPrior` — blended prior with `ρ = 1 − 1/(1 + n/(φ²·τ))`, equals 0.5 at `n = φ²·τ`
- `FederatedDelta` — sufficient statistics for privacy-preserving federated aggregation
- Differential privacy support: optional Laplace noise on federated deltas
- φ constraint enforcement after each federated aggregation step
- `Phase4GoldenFusion.commit_session()` — pushes session evidence deltas to hierarchical model

### Added — NamoNexusEngine
- `NamoNexusEngine` — unified facade over the complete Phase 1–4 stack
- `NamoNexusEngine.update_batch()` — convenience method for multi-observation updates
- `NamoNexusEngine.session_summary()` — JSON-serialisable full-stack session digest

---

## [3.0.0] — Phase 3: Production Grade

### Added — Feature 3.1: Drift Detection (Patent Claim 11)
- `DriftDetector` — per-modality Page-Hinkley CUSUM with φ-initialised thresholds
- Detection threshold `h = φ²`, sensitivity `δ = 1/φ²`
- Adaptive `h_eff = h · (1 + U · φ · scale)` — widens with posterior uncertainty
- `DriftEvent` with severity levels: WARNING and ALARM
- `Phase3GoldenFusion.drift_events()`, `has_drift_alarm`, `drift_summary()`

### Added — Feature 3.2: Streaming Inference (Patent Claim 12)
- `StreamingPipeline` — async-first streaming inference with sync `run_sync()` wrapper
- `SlidingWindowAnalyzer` — O(1) rolling statistics (mean score, risk trend, modality mix)
- `DeliveryLedger` — write-ahead log for at-least-once delivery guarantee
- `BackpressureController` — bounded queue depth, pauses producers when full
- Connector protocol: `InMemoryConnector`, `KafkaConnector`, `WebSocketConnector`
- `StreamResult` — per-observation result with window statistics

---

## [2.0.0] — Phase 2: Patent Core Features

### Added — Feature 2.1: Modality Auto-Calibration (Patent Claim 8)
- `ModalityCalibrator` — per-modality Beta trust posterior initialised at `τ_α/τ_β = φ`
- Consistency-based trust update: `consistency_m = 1 − |score_m − fused_before|`
- `effective_confidence = raw_confidence × trust_mean`

### Added — Feature 2.2: Sensor Trust Scoring (Patent Claim 9)
- `SensorTrustScorer` — rolling Page-Hinkley anomaly detection per sensor
- Automatic blacklisting on sustained anomaly + cooldown + reinstatement
- Trust event callbacks for monitoring integration

### Added — Feature 2.3: Online Hyperparameter Optimisation (Patent Claim 10)
- `OnlineHyperparamOptimizer` — gradient-free evolutionary search preserving φ constraint
- Self-supervised composite feedback score (calibration, sharpness, consistency, sensitivity)
- Hot-swap config: replaces `FusionConfig` without resetting posterior state

---

## [1.0.0] — Phase 1: Scientific Foundation

### Added — Feature 1.1: Temporal Bayesian Filtering (Patent Claim 6)
- `TemporalBayesianFilter` — exponential forgetting applied to accumulated evidence only
- Default decay factor `λ₀ = 1/φ ≈ 0.618` (Golden Ratio reciprocal)
- Prior `α₀/β₀ = φ` preserved exactly under any decay schedule
- Adaptive mode: λ adjusts to score velocity
- Time-based mode: additional decay proportional to elapsed wall-clock time

### Added — Feature 1.2: Empirical Prior Learning (Patent Claim 7)
- `EmpiricalPriorLearner` — MLE with Golden Ratio regularisation `reg · (α/β − φ)²`
- `SyntheticSessionGenerator` — multi-modality synthetic data with subject profiles
- `LearnedPrior` — serialisable personalised prior for session warm-start

---

## [0.3.0] — v3.0 Foundation Engine

### Added — Base GoldenBayesianFusion (Patent Claims 1–5)
- `GoldenBayesianFusion` — Beta-conjugate posterior with `α₀/β₀ = φ`
- Confidence-to-trials mapping: `n = max_trials · c^k`
- Risk classification: LOW / MEDIUM / HIGH / CRITICAL
- Credible interval via `scipy.stats.beta.ppf`
- Deception probability: `P(score > threshold)` under posterior
- Full state serialisation: `get_state()` / `load_state()`
- `FusionConfig` dataclass with full validation
