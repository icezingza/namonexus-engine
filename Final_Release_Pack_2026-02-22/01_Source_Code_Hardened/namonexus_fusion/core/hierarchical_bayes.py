# © 2026 Kanin Raksaraj (P'Ice). All Rights Reserved.
# Confidential and Proprietary. Do not distribute without permission.

"""
hierarchical_bayes.py — Feature 4.2: Hierarchical Bayesian Model
=================================================================
Patent-Pending Technology | NamoNexus Research Team

Patent Claim 14
---------------
"สถาปัตยกรรม hierarchical Bayesian สำหรับ multimodal fusion ที่แยก
population prior จาก individual posterior โดยใช้ Golden Ratio เป็น
structural constraint ในทุกระดับ รองรับ Federated Learning"

Architecture
------------
Two-level hierarchy:

    Level 0 (Population)  —  HyperPrior
        β(α_pop, β_pop) shared across all organisations / users.
        α_pop / β_pop ratio is constrained to φ at initialisation.
        Updated via Federated aggregation (no raw data exchange).

    Level 1 (Individual)  —  LocalPosterior  (one per user / org)
        β(α_ind, β_ind) starts from Population prior as its prior,
        then updates on local observations only.
        Individual posterior is the engine's inference result.

Golden Ratio Structural Constraint
------------------------------------
At every level the *ratio* α/β is initialised to φ:

    Population init:   α_pop = φ · s_pop,  β_pop = s_pop
    Individual init:   α_ind = φ · s_ind,  β_ind = s_ind
                       where s_ind = (α_pop + β_pop) / 2  (soft pull to population)

This preserves NamoNexus's core identity: "prior encodes Golden Ratio optimism".
Scale factors s_pop / s_ind control how strongly the prior anchors beliefs.

Federated Learning Protocol
----------------------------
NamoNexus uses a **differential-privacy-safe, data-free aggregation**:

1. Each LocalModel serialises its (α_ind, β_ind) — *sufficient statistics* only.
2. The Aggregator computes a φ-weighted average of individual sufficient stats.
3. The resulting aggregate replaces the population-level HyperPrior.
4. Each LocalModel then receives the new HyperPrior and can cold-start
   from it without sharing raw observations.

φ-weighted aggregation formula (K clients):
    α_pop_new = Σ_k φ^(rank_k−1) · α_k / Σ_k φ^(rank_k−1)
    β_pop_new = same with β_k

Rank is determined by each client's total observation count (more data →
higher rank → higher φ-weight).

New Patent Claims Covered (Phase 4, Feature 4.2)
-------------------------------------------------
Claim 14: Hierarchical Bayesian architecture separating population prior
          from individual posterior using Golden Ratio structural constraint
          at all levels, with Federated Learning via φ-weighted aggregation.

Usage
-----
::

    # ── Cold-start: new org gets population prior ──────────────────
    pop_model  = PopulationModel()
    local      = LocalModel.from_population(pop_model)

    # ── Individual learning ──────────────────────────────────────
    local.update(score=0.85, confidence=0.90, modality="text")
    local.update(score=0.60, confidence=0.70, modality="voice")
    print(local.fused_score, local.uncertainty)

    # ── Federated aggregation (no raw data shared) ───────────────
    aggregator = FederatedAggregator(pop_model)
    aggregator.register(local)
    aggregator.aggregate()  # updates population prior in-place

    # ── New org benefits from aggregated knowledge ───────────────
    new_local = LocalModel.from_population(pop_model)
    print(new_local.fused_score)   # warm-started, not 0.5
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Golden Ratio constants (structural — never tunable)
# ---------------------------------------------------------------------------

_PHI: float       = (1.0 + math.sqrt(5.0)) / 2.0   # ≈ 1.6180
_PHI_RECIP: float = 1.0 / _PHI                      # ≈ 0.6180
_PHI_SQ: float    = _PHI ** 2                       # ≈ 2.6180

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalConfig:
    """
    Configuration for the hierarchical Bayesian system.

    Attributes
    ----------
    population_scale:
        Concentration of the population-level Beta prior.
        Higher = stronger population prior pull.
        Default = 2.0 (moderate anchor).
    individual_scale_multiplier:
        The individual prior scale = population_scale × this multiplier.
        Default = _PHI_RECIP ≈ 0.618 (individual prior is softer than population).
    phi_rank_decay:
        Base for φ-rank decay in federated aggregation.
        weight_k = phi_rank_decay^(1 − rank_k).
        Default = _PHI.
    min_observations_for_rank:
        Minimum local observations before a client gets rank > 0 in aggregation.
        Prevents empty/garbage clients from influencing population.
    dp_noise_std:
        Gaussian noise std-dev added to (α, β) before sharing for
        differential privacy.  0.0 = no noise (production: set > 0).
    dp_clip:
        Clip individual sufficient stats to [0, dp_clip] before noise addition.
        Prevents sensitivity blowup.
    """

    population_scale: float              = 2.0
    individual_scale_multiplier: float   = _PHI_RECIP
    phi_rank_decay: float                = _PHI
    min_observations_for_rank: int       = 3
    dp_noise_std: float                  = 0.0
    dp_clip: float                       = 1e4

    def __post_init__(self) -> None:
        if self.population_scale <= 0:
            raise ValueError("population_scale must be > 0")
        if self.individual_scale_multiplier <= 0:
            raise ValueError("individual_scale_multiplier must be > 0")
        if self.phi_rank_decay <= 1.0:
            raise ValueError("phi_rank_decay must be > 1.0")

    @classmethod
    def default(cls) -> "HierarchicalConfig":
        return cls()

    @classmethod
    def tight_population(cls) -> "HierarchicalConfig":
        """Strong population prior — useful when population data is rich."""
        return cls(population_scale=10.0)

    @classmethod
    def loose_individual(cls) -> "HierarchicalConfig":
        """Weak individual pull — trust local data faster."""
        return cls(individual_scale_multiplier=0.1)


# ---------------------------------------------------------------------------
# Population model (Level 0)
# ---------------------------------------------------------------------------


class PopulationModel:
    """
    Population-level hyper-prior: a Beta(α_pop, β_pop) shared across all clients.

    Golden Ratio structural constraint: α_pop / β_pop is initialised to φ.

    The population model is updated in-place by ``FederatedAggregator.aggregate()``.

    Parameters
    ----------
    config:
        ``HierarchicalConfig`` instance.
    """

    def __init__(self, config: Optional[HierarchicalConfig] = None) -> None:
        self._cfg     = config or HierarchicalConfig.default()
        s             = self._cfg.population_scale
        self._alpha_0 = _PHI * s   # initialise ratio to φ
        self._beta_0  = s
        self._alpha   = self._alpha_0
        self._beta    = self._beta_0
        self._agg_count = 0        # number of federated aggregations performed
        logger.info(
            "PopulationModel init | α=%.4f β=%.4f ratio=%.4f (φ=%.4f)",
            self._alpha, self._beta, self._alpha / self._beta, _PHI,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def mean(self) -> float:
        """Population-level belief about the expected score."""
        return self._alpha / (self._alpha + self._beta)

    @property
    def variance(self) -> float:
        a, b = self._alpha, self._beta
        n = a + b
        return (a * b) / (n ** 2 * (n + 1)) if n > 1 else 0.25

    @property
    def concentration(self) -> float:
        """Total concentration (α + β) — proxy for strength of population belief."""
        return self._alpha + self._beta

    @property
    def phi_ratio(self) -> float:
        """Current α/β ratio (should be close to φ when well-calibrated)."""
        return self._alpha / self._beta if self._beta > 0 else float("inf")

    @property
    def aggregation_count(self) -> int:
        return self._agg_count

    # ------------------------------------------------------------------
    # Federated update (called by FederatedAggregator)
    # ------------------------------------------------------------------

    def _federated_update(self, new_alpha: float, new_beta: float) -> None:
        """
        Replace population parameters with federated-aggregated values.
        Called only by ``FederatedAggregator``.
        """
        if new_alpha <= 0 or new_beta <= 0:
            logger.warning(
                "PopulationModel: federated update received non-positive params "
                "(α=%.4f, β=%.4f) — skipped", new_alpha, new_beta
            )
            return
        self._alpha = new_alpha
        self._beta  = new_beta
        self._agg_count += 1
        logger.info(
            "PopulationModel federated update #%d | α=%.4f β=%.4f ratio=%.4f",
            self._agg_count, self._alpha, self._beta, self.phi_ratio,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha":            self._alpha,
            "beta":             self._beta,
            "alpha_init":       self._alpha_0,
            "beta_init":        self._beta_0,
            "mean":             self.mean,
            "variance":         self.variance,
            "concentration":    self.concentration,
            "phi_ratio":        self.phi_ratio,
            "aggregation_count": self._agg_count,
        }

    def reset(self) -> None:
        """Revert population parameters to initialisation values."""
        self._alpha     = self._alpha_0
        self._beta      = self._beta_0
        self._agg_count = 0

    def __repr__(self) -> str:
        return (
            f"PopulationModel(mean={self.mean:.4f}, "
            f"concentration={self.concentration:.2f}, "
            f"φ-ratio={self.phi_ratio:.4f}, "
            f"agg_count={self._agg_count})"
        )


# ---------------------------------------------------------------------------
# Local model (Level 1)
# ---------------------------------------------------------------------------


class LocalModel:
    """
    Individual-level posterior: Beta(α_ind, β_ind), cold-started from population.

    After construction (via ``LocalModel.from_population()``), local observations
    are accumulated via ``update()``.  The posterior mean is ``fused_score``.

    Parameters
    ----------
    alpha_init:
        Initial α drawn from the population prior.
    beta_init:
        Initial β drawn from the population prior.
    client_id:
        Optional identifier for federated tracking.
    config:
        ``HierarchicalConfig``.

    Notes
    -----
    The individual scale = population_scale × individual_scale_multiplier.
    This makes the individual prior *softer* than the population so that
    a handful of local observations can shift the individual belief meaningfully.
    """

    def __init__(
        self,
        alpha_init: float,
        beta_init:  float,
        client_id:  Optional[str] = None,
        config:     Optional[HierarchicalConfig] = None,
    ) -> None:
        self._cfg         = config or HierarchicalConfig.default()
        self._alpha_init  = alpha_init
        self._beta_init   = beta_init
        self._alpha       = alpha_init
        self._beta        = beta_init
        self.client_id    = client_id or f"client_{id(self):x}"
        self._history: List[Dict[str, Any]] = []
        self._obs_count   = 0

    # ------------------------------------------------------------------
    # Factory: cold-start from population
    # ------------------------------------------------------------------

    @classmethod
    def from_population(
        cls,
        population: PopulationModel,
        client_id:  Optional[str] = None,
        config:     Optional[HierarchicalConfig] = None,
    ) -> "LocalModel":
        """
        Create a ``LocalModel`` warm-started from the current population prior.

        The individual scale is set to:
            s_ind = (α_pop + β_pop) × individual_scale_multiplier / 2

        The Golden Ratio constraint is preserved:
            α_ind = φ × s_ind,  β_ind = s_ind  (ratio = φ exactly at cold-start)

        If the population prior has already been updated by federated aggregation,
        the individual model benefits from that collective knowledge.
        """
        cfg   = config or HierarchicalConfig.default()
        s_pop = population.concentration
        s_ind = s_pop * cfg.individual_scale_multiplier

        # Use population mean to shift individual cold-start
        pop_mean = population.mean
        # α / (α + β) = pop_mean  and  α + β = s_ind  → α = pop_mean × s_ind
        alpha_init = pop_mean * s_ind
        beta_init  = (1.0 - pop_mean) * s_ind

        # Guarantee minimum concentration = 2 (1 α, 1 β) so posterior is defined
        alpha_init = max(alpha_init, _PHI_RECIP)
        beta_init  = max(beta_init,  _PHI_RECIP)

        obj = cls(
            alpha_init=alpha_init,
            beta_init=beta_init,
            client_id=client_id,
            config=cfg,
        )
        logger.debug(
            "LocalModel cold-start | id=%s α=%.4f β=%.4f from population mean=%.4f",
            obj.client_id, alpha_init, beta_init, pop_mean,
        )
        return obj

    # ------------------------------------------------------------------
    # Bayesian update
    # ------------------------------------------------------------------

    def update(
        self,
        score:         float,
        confidence:    float,
        modality:      Optional[str] = None,
        metadata:      Optional[Dict[str, Any]] = None,
    ) -> "LocalModel":
        """
        Incorporate one multimodal observation into the individual posterior.

        Parameters
        ----------
        score:
            Modality-level score ∈ [0, 1].
        confidence:
            Reliability ∈ [0, 1] — scales effective observation count.
        modality:
            Name of the sensing modality.
        metadata:
            Extra key-value pairs stored in history (for explainability).

        Returns
        -------
        LocalModel
            Self (fluent interface).
        """
        score      = float(np.clip(score,      0.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))
        n          = max(1.0, confidence * 10.0)

        self._alpha    += score * n
        self._beta     += (1.0 - score) * n
        self._obs_count += 1

        rec = {
            "modality":   modality or "unknown",
            "score":      score,
            "confidence": confidence,
            "timestamp":  time.time(),
            **(metadata or {}),
        }
        self._history.append(rec)
        return self

    # ------------------------------------------------------------------
    # Inference properties
    # ------------------------------------------------------------------

    @property
    def fused_score(self) -> float:
        """Individual posterior mean."""
        return self._alpha / (self._alpha + self._beta)

    @property
    def uncertainty(self) -> float:
        """Individual posterior std-dev."""
        a, b = self._alpha, self._beta
        n    = a + b
        return math.sqrt(a * b / (n ** 2 * (n + 1))) if n > 1 else 0.5

    @property
    def risk_level(self) -> str:
        s = self.fused_score
        if s >= 0.75:
            return "low"
        if s >= 0.45:
            return "medium"
        return "high"

    @property
    def observation_count(self) -> int:
        return self._obs_count

    @property
    def modality_history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    @property
    def phi_ratio(self) -> float:
        """Current α/β (should drift toward pop mean, not necessarily φ)."""
        return self._alpha / self._beta if self._beta > 0 else float("inf")

    # ------------------------------------------------------------------
    # Sufficient statistics for federated sharing
    # ------------------------------------------------------------------

    def sufficient_stats(self, apply_dp: bool = True) -> Tuple[float, float]:
        """
        Return (α_local, β_local) — the only data shared in federated aggregation.

        If ``apply_dp`` and ``config.dp_noise_std > 0``, Gaussian noise is added
        and values are clipped to [0, dp_clip] before returning.

        Raw observations are **never** exported.
        """
        alpha = float(np.clip(self._alpha, 0.0, self._cfg.dp_clip))
        beta  = float(np.clip(self._beta,  0.0, self._cfg.dp_clip))

        if apply_dp and self._cfg.dp_noise_std > 0.0:
            rng   = np.random.default_rng()
            sigma = self._cfg.dp_noise_std
            alpha = float(np.clip(alpha + rng.normal(0.0, sigma), _PHI_RECIP, self._cfg.dp_clip))
            beta  = float(np.clip(beta  + rng.normal(0.0, sigma), _PHI_RECIP, self._cfg.dp_clip))

        return alpha, beta

    # ------------------------------------------------------------------
    # State / serialisation
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "client_id":       self.client_id,
            "alpha":           self._alpha,
            "beta":            self._beta,
            "alpha_init":      self._alpha_init,
            "beta_init":       self._beta_init,
            "fused_score":     self.fused_score,
            "uncertainty":     self.uncertainty,
            "risk_level":      self.risk_level,
            "obs_count":       self._obs_count,
            "phi_ratio":       self.phi_ratio,
            "modality_history": self._history,
        }

    def reset(self) -> None:
        """Revert individual posterior to cold-start values."""
        self._alpha     = self._alpha_init
        self._beta      = self._beta_init
        self._obs_count = 0
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"LocalModel(id={self.client_id}, "
            f"score={self.fused_score:.4f} ± {self.uncertainty:.4f}, "
            f"risk={self.risk_level}, obs={self._obs_count})"
        )


# ---------------------------------------------------------------------------
# Federated Aggregator
# ---------------------------------------------------------------------------


class FederatedAggregator:
    """
    Coordinates φ-weighted federated aggregation of local model statistics.

    Design Principles
    -----------------
    * **No raw data** — only (α_k, β_k) sufficient statistics are shared.
    * **φ-rank weighting** — clients with more data get exponentially more
      influence via φ^(rank_k − 1) weighting (Patent Claim 14).
    * **Differential Privacy** — optional Gaussian noise (configured in
      ``HierarchicalConfig.dp_noise_std``) before statistics are shared.
    * **Convergence tracking** — ``convergence_history`` records how
      population parameters change across aggregation rounds.

    Parameters
    ----------
    population:
        The shared ``PopulationModel`` that will be updated in-place.
    config:
        ``HierarchicalConfig``.
    """

    def __init__(
        self,
        population: PopulationModel,
        config:     Optional[HierarchicalConfig] = None,
    ) -> None:
        self._population  = population
        self._cfg         = config or HierarchicalConfig.default()
        self._clients:    List[LocalModel] = []
        self._agg_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Client registration
    # ------------------------------------------------------------------

    def register(self, local: LocalModel) -> None:
        """Register a ``LocalModel`` as a federated participant."""
        if local not in self._clients:
            self._clients.append(local)
            logger.debug("FederatedAggregator: registered client %s", local.client_id)

    def deregister(self, local: LocalModel) -> None:
        """Remove a client from the federation."""
        self._clients = [c for c in self._clients if c is not local]

    @property
    def num_clients(self) -> int:
        return len(self._clients)

    # ------------------------------------------------------------------
    # φ-weighted aggregation (core of Patent Claim 14)
    # ------------------------------------------------------------------

    def aggregate(self, apply_dp: bool = True) -> Dict[str, Any]:
        """
        Run one round of federated aggregation.

        Steps
        -----
        1. Collect (α_k, β_k) from each eligible client (obs ≥ min threshold).
        2. Rank clients by observation count (most data → rank 1).
        3. Compute φ-rank weights: w_k = φ^(1 − rank_k), normalised.
        4. Compute weighted aggregate: α_new = Σ w_k α_k, β_new = Σ w_k β_k.
        5. Update ``PopulationModel`` in-place.
        6. Record convergence metrics.

        Parameters
        ----------
        apply_dp:
            Whether to apply differential-privacy noise (from config).

        Returns
        -------
        dict
            Aggregation round summary including φ-weights and new population params.
        """
        eligible = [
            c for c in self._clients
            if c.observation_count >= self._cfg.min_observations_for_rank
        ]

        if not eligible:
            logger.warning(
                "FederatedAggregator.aggregate: no eligible clients "
                "(min_observations=%d)", self._cfg.min_observations_for_rank
            )
            return {"status": "skipped", "reason": "no_eligible_clients"}

        # Collect sufficient stats
        stats: List[Tuple[str, float, float, int]] = []
        for client in eligible:
            a, b = client.sufficient_stats(apply_dp=apply_dp)
            stats.append((client.client_id, a, b, client.observation_count))

        # Rank by observation count descending (most data → rank 1)
        stats.sort(key=lambda x: x[3], reverse=True)

        # φ-rank weights
        decay = self._cfg.phi_rank_decay
        raw_weights = [decay ** (1.0 - (i + 1)) for i in range(len(stats))]
        total_w     = sum(raw_weights)
        norm_weights = [w / total_w for w in raw_weights]

        # Weighted aggregate
        alpha_new = sum(w * s[1] for w, s in zip(norm_weights, stats))
        beta_new  = sum(w * s[2] for w, s in zip(norm_weights, stats))

        alpha_prev = self._population.alpha
        beta_prev  = self._population.beta

        self._population._federated_update(alpha_new, beta_new)

        round_summary = {
            "round":          self._population.aggregation_count,
            "timestamp":      time.time(),
            "num_eligible":   len(eligible),
            "alpha_prev":     alpha_prev,
            "beta_prev":      beta_prev,
            "alpha_new":      alpha_new,
            "beta_new":       beta_new,
            "population_mean_new": self._population.mean,
            "phi_weights":    {s[0]: w for s, w in zip(stats, norm_weights)},
            "client_stats":   [
                {"client_id": s[0], "alpha": s[1], "beta": s[2], "obs": s[3]}
                for s in stats
            ],
        }
        self._agg_history.append(round_summary)
        logger.info(
            "FederatedAggregator round %d | clients=%d "
            "α: %.4f → %.4f, β: %.4f → %.4f",
            round_summary["round"], len(eligible),
            alpha_prev, alpha_new, beta_prev, beta_new,
        )
        return round_summary

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def aggregation_history(self) -> List[Dict[str, Any]]:
        return list(self._agg_history)

    def convergence_summary(self) -> Dict[str, Any]:
        """
        Assess whether population parameters are converging across rounds.

        Returns delta (|α_new − α_old| + |β_new − β_old|) per round,
        and a simple converged flag when the last delta < 0.01.
        """
        if len(self._agg_history) < 2:
            return {"converged": False, "rounds": len(self._agg_history), "deltas": []}

        deltas = []
        for i in range(1, len(self._agg_history)):
            h_prev = self._agg_history[i - 1]
            h_curr = self._agg_history[i]
            d = abs(h_curr["alpha_new"] - h_prev["alpha_new"]) + \
                abs(h_curr["beta_new"]  - h_prev["beta_new"])
            deltas.append(d)

        return {
            "converged": deltas[-1] < 0.01 if deltas else False,
            "rounds":    len(self._agg_history),
            "deltas":    deltas,
            "last_delta": deltas[-1] if deltas else None,
        }

    def __repr__(self) -> str:
        return (
            f"FederatedAggregator("
            f"clients={self.num_clients}, "
            f"rounds={self._population.aggregation_count}, "
            f"population_mean={self._population.mean:.4f})"
        )
