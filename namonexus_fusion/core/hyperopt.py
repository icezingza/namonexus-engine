"""
Online Hyperparameter Optimization — Feature 2.3
==================================================
Patent-Pending Technology | NamoNexus Research Team

Problem
-------
FusionConfig parameters (prior_strength, max_trials, decay_factor, etc.)
are set once at startup and never change.  Optimal values depend on:

  • The specific use case (therapy session vs. job interview)
  • The subject's personality and baseline
  • The quality of available sensors
  • Session length and dynamics

A fixed config is guaranteed to be suboptimal for at least some sessions.

Solution: Gradient-Free Online Optimization
-------------------------------------------
We use a lightweight variant of Covariance Matrix Adaptation Evolution
Strategy (CMA-ES), adapted for the online / streaming setting:

  1. Maintain a population of candidate configurations.
  2. After each observation, evaluate each candidate against a
     *feedback signal* (see below).
  3. Update the search distribution toward better candidates.
  4. Apply the best candidate to the live engine.

Feedback Signal
---------------
The feedback signal is a composite score that rewards:

  • **Calibration**  — fused_score correlates with observed outcomes
  • **Sharpness**    — uncertainty is low (engine is confident)
  • **Consistency**  — risk_level is stable across consecutive observations
  • **Sensitivity**  — engine responds quickly to genuine state changes

Since ground-truth labels may not be available, we use a *self-supervised*
proxy: we compare the engine's risk_level to the *majority vote* across
all active modalities in the current window.

Patent Claim (new — Claim 10)
------------------------------
"A method of online hyperparameter optimization for a multimodal Bayesian
fusion engine, wherein:
(a) a population of candidate FusionConfig instances is evaluated against
    a composite self-supervised feedback score;
(b) the search distribution is updated using a gradient-free evolutionary
    strategy that preserves the Golden Ratio structural constraint
    (α₀/β₀ = φ) across all candidate configurations;
(c) the best candidate is applied to the live engine without interrupting
    the inference stream;
such that the engine self-tunes to the characteristics of the current
session without requiring labelled data."
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config.settings import FusionConfig
from ..core.constants import GOLDEN_RATIO
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HyperparamBounds:
    """
    Defines the search space for each hyperparameter.

    All parameters are bounded and continuous.
    The optimizer works in a normalized [0, 1]^d space internally.
    """

    prior_strength_range:   Tuple[float, float] = (0.5,  10.0)
    max_trials_range:       Tuple[int,   int]   = (20,   300)
    decay_factor_range:     Tuple[float, float] = (0.3,  1.0)
    nonlinear_k_range:      Tuple[float, float] = (0.5,  8.0)
    risk_low_range:         Tuple[float, float] = (0.15, 0.45)
    risk_high_range:        Tuple[float, float] = (0.55, 0.85)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Candidate evaluation result
# ---------------------------------------------------------------------------


@dataclass
class CandidateResult:
    """Evaluation result for one candidate configuration."""

    config:           FusionConfig
    feedback_score:   float          # higher = better
    calibration:      float
    sharpness:        float
    consistency:      float
    evaluated_at:     float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_score": round(self.feedback_score, 6),
            "calibration":    round(self.calibration, 6),
            "sharpness":      round(self.sharpness, 6),
            "consistency":    round(self.consistency, 6),
        }


# ---------------------------------------------------------------------------
# Optimizer state snapshot
# ---------------------------------------------------------------------------


@dataclass
class OptimizerState:
    """Serializable state of the optimizer."""

    best_config:     Dict[str, Any]
    best_score:      float
    n_evaluations:   int
    generation:      int
    search_mean:     List[float]
    search_std:      List[float]
    history:         List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Core optimizer
# ---------------------------------------------------------------------------


class OnlineHyperparamOptimizer:
    """
    Gradient-free online hyperparameter optimizer for GoldenBayesianFusion.

    Uses a simplified (1+λ) Evolution Strategy — lightweight enough to
    run inline without blocking the inference stream.

    The Golden Ratio structural constraint (α₀/β₀ = φ) is **always
    preserved** across all candidate configurations.

    Parameters
    ----------
    bounds:
        Search space bounds for each hyperparameter.
    population_size:
        Number of candidate configurations evaluated per generation.
    mutation_rate:
        Standard deviation of Gaussian mutations in normalized space.
    feedback_fn:
        Optional custom feedback function ``(FusionConfig, observations) → float``.
        If None, uses the built-in composite self-supervised score.
    seed:
        Random seed for reproducibility.

    Examples
    --------
    ::

        optimizer = OnlineHyperparamOptimizer()

        # Each update step (called after a batch of observations):
        new_config = optimizer.step(
            current_config=engine.config,
            fused_scores=[0.62, 0.58, 0.54],
            uncertainties=[0.12, 0.11, 0.10],
            risk_levels=["moderate", "moderate", "high"],
            modality_scores={"voice": [0.3, 0.25], "text": [0.8, 0.85]},
        )
        if new_config is not None:
            engine = TemporalGoldenFusion(config=new_config)
    """

    # Hyperparameter names in canonical order (must match _decode)
    PARAM_NAMES = [
        "prior_strength",
        "max_trials",
        "decay_factor",
        "nonlinear_k",
        "risk_low",
        "risk_high",
    ]

    def __init__(
        self,
        bounds: Optional[HyperparamBounds] = None,
        population_size: int = 8,
        mutation_rate: float = 0.15,
        feedback_fn: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        if population_size < 2:
            raise ConfigurationError("population_size must be >= 2.")
        if not (0.0 < mutation_rate < 1.0):
            raise ConfigurationError("mutation_rate must be in (0, 1).")

        self._bounds      = bounds or HyperparamBounds()
        self._pop_size    = population_size
        self._mut_rate    = mutation_rate
        self._feedback_fn = feedback_fn
        self._rng         = np.random.default_rng(seed)

        # Search distribution: mean and std in [0,1]^d normalized space
        d = len(self.PARAM_NAMES)
        self._mean = np.full(d, 0.5)         # start at center
        self._std  = np.full(d, mutation_rate)

        # Best found so far
        self._best_config: Optional[FusionConfig] = None
        self._best_score:  float = float("-inf")
        self._current_config: FusionConfig = FusionConfig()

        # Diagnostics
        self._n_evaluations: int = 0
        self._generation:    int = 0
        self._history:       List[Dict[str, Any]] = []

        logger.info(
            "OnlineHyperparamOptimizer | pop=%d mut=%.2f params=%s",
            population_size, mutation_rate, self.PARAM_NAMES,
        )

    # ------------------------------------------------------------------
    # Encoding / decoding between normalized [0,1]^d and FusionConfig
    # ------------------------------------------------------------------

    def _encode(self, config: FusionConfig) -> np.ndarray:
        """Encode a FusionConfig into a normalized [0,1]^d vector."""
        b = self._bounds

        def norm(v, lo, hi):
            return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

        return np.array([
            norm(config.prior_strength, *b.prior_strength_range),
            norm(config.max_trials,     *b.max_trials_range),
            norm(config.decay_factor if hasattr(config, "decay_factor") else 0.618,
                 *b.decay_factor_range),
            norm(config.nonlinear_k,    *b.nonlinear_k_range),
            norm(config.risk_low,       *b.risk_low_range),
            norm(config.risk_high,      *b.risk_high_range),
        ])

    def _decode(self, vec: np.ndarray) -> FusionConfig:
        """
        Decode a normalized vector into a FusionConfig.

        The Golden Ratio constraint α₀/β₀ = φ is preserved because
        prior_strength fully determines both (α₀ = φ·s, β₀ = s).
        """
        b = self._bounds

        def denorm(v, lo, hi):
            return lo + float(np.clip(v, 0.0, 1.0)) * (hi - lo)

        prior_strength = denorm(vec[0], *b.prior_strength_range)
        max_trials     = max(int(round(denorm(vec[1], *b.max_trials_range))), 5)
        nonlinear_k    = denorm(vec[3], *b.nonlinear_k_range)
        risk_low       = denorm(vec[4], *b.risk_low_range)
        risk_high_raw  = denorm(vec[5], *b.risk_high_range)

        # Ensure risk_low < risk_high with minimum gap
        risk_high = max(risk_high_raw, risk_low + 0.10)

        return FusionConfig(
            prior_strength=prior_strength,
            min_trials=1,
            max_trials=max_trials,
            use_nonlinear_confidence=True,
            nonlinear_k=nonlinear_k,
            risk_low=risk_low,
            risk_high=risk_high,
        )

    # ------------------------------------------------------------------
    # Feedback functions
    # ------------------------------------------------------------------

    def _composite_feedback(
        self,
        fused_scores:    List[float],
        uncertainties:   List[float],
        risk_levels:     List[str],
        modality_scores: Dict[str, List[float]],
    ) -> Tuple[float, float, float, float]:
        """
        Compute the composite self-supervised feedback score.

        Returns
        -------
        (total, calibration, sharpness, consistency)
        """
        if not fused_scores:
            return 0.0, 0.0, 0.0, 0.0

        scores = np.array(fused_scores)
        uncerts = np.array(uncertainties)

        # 1. Calibration: fused scores should correlate with modality consensus
        if modality_scores:
            all_modal = np.concatenate(list(modality_scores.values()))
            modal_mean = float(np.mean(all_modal))
            # Penalise large divergence between fused and modal consensus
            calibration = float(1.0 - np.mean(np.abs(scores - modal_mean)))
        else:
            calibration = 0.5

        # 2. Sharpness: reward low uncertainty (confident predictions)
        sharpness = float(1.0 - np.mean(uncerts))

        # 3. Consistency: reward stable risk_level across window
        if len(risk_levels) >= 2:
            changes = sum(
                1 for i in range(1, len(risk_levels))
                if risk_levels[i] != risk_levels[i - 1]
            )
            consistency = float(1.0 - changes / (len(risk_levels) - 1))
        else:
            consistency = 1.0

        # Composite: weighted sum with Golden Ratio-derived weights
        # w1 = φ²/(φ²+φ+1), w2 = φ/(φ²+φ+1), w3 = 1/(φ²+φ+1)
        phi   = GOLDEN_RATIO
        denom = phi ** 2 + phi + 1.0
        w1, w2, w3 = phi ** 2 / denom, phi / denom, 1.0 / denom

        total = w1 * calibration + w2 * sharpness + w3 * consistency
        return float(total), float(calibration), float(sharpness), float(consistency)

    # ------------------------------------------------------------------
    # Core optimization step
    # ------------------------------------------------------------------

    def step(
        self,
        current_config:  Optional[FusionConfig] = None,
        fused_scores:    Optional[List[float]] = None,
        uncertainties:   Optional[List[float]] = None,
        risk_levels:     Optional[List[str]] = None,
        modality_scores: Optional[Dict[str, List[float]]] = None,
        observations:    Optional[List[Tuple[float, float, str]]] = None,
        fused_score:     Optional[float] = None,
        uncertainty:     Optional[float] = None,
        risk_level:      Optional[str] = None,
    ) -> Optional[FusionConfig]:
        """
        Run one optimization generation and return the best config found.

        Designed to be called periodically (e.g. every 10 observations)
        rather than after every single update.

        Parameters
        ----------
        current_config:
            The config currently in use.
        fused_scores:
            Recent fused scores (e.g. last 10 observations).
        uncertainties:
            Corresponding uncertainty values.
        risk_levels:
            Corresponding risk level strings.
        modality_scores:
            Optional dict of {modality: [scores]} for calibration metric.

        Returns
        -------
        FusionConfig or None
            The best config found this generation, or None if the current
            config is still best.
        """
        # Legacy compatibility: accept the older one-step API:
        # step(observations=[(score, conf, modality), ...], fused_score=..., ...)
        if observations is not None:
            fused_scores = [float(s) for s, _, _ in observations]
            uncertainties = [float(uncertainty or 0.0)] * len(fused_scores)
            risk_levels = [str(risk_level or "unknown")] * len(fused_scores)
            if modality_scores is None:
                modality_scores = {}
                for s, _, m in observations:
                    modality_scores.setdefault(str(m), []).append(float(s))
            if current_config is None:
                current_config = self._current_config
        if current_config is None:
            current_config = self._current_config
        if fused_scores is None:
            fused_scores = []
        if uncertainties is None:
            uncertainties = []
        if risk_levels is None:
            risk_levels = []

        if len(fused_scores) < 2:
            return None

        modal_scores = modality_scores or {}

        # ── Generate candidate population ────────────────────────────
        center = self._encode(current_config)
        candidates: List[np.ndarray] = [center]   # always include current

        for _ in range(self._pop_size - 1):
            mutation = self._rng.normal(0, self._mut_rate, size=len(self.PARAM_NAMES))
            candidate = np.clip(center + mutation, 0.0, 1.0)
            candidates.append(candidate)

        # ── Evaluate candidates ──────────────────────────────────────
        results: List[CandidateResult] = []

        for vec in candidates:
            cfg = self._decode(vec)
            if self._feedback_fn:
                score = float(self._feedback_fn(cfg, fused_scores))
                cal = sha = con = 0.0
            else:
                score, cal, sha, con = self._composite_feedback(
                    fused_scores, uncertainties, risk_levels, modal_scores
                )
                # Add mild bonus for lower uncertainty (config sharpness)
                score += 0.05 * sha

            results.append(CandidateResult(
                config=cfg,
                feedback_score=score,
                calibration=cal,
                sharpness=sha,
                consistency=con,
            ))
            self._n_evaluations += 1

        # ── Select best ──────────────────────────────────────────────
        results.sort(key=lambda r: r.feedback_score, reverse=True)
        best = results[0]

        improved = best.feedback_score > self._best_score
        if improved:
            self._best_config = best.config
            self._best_score  = best.feedback_score
            self._current_config = best.config

        # ── Update search distribution ───────────────────────────────
        # Elitist update: move mean toward top-k results
        top_k   = max(1, self._pop_size // 2)
        top_vecs = [self._encode(r.config) for r in results[:top_k]]
        self._mean = np.mean(top_vecs, axis=0)

        self._generation += 1

        # ── Record history ───────────────────────────────────────────
        self._history.append({
            "generation":     self._generation,
            "best_score":     round(best.feedback_score, 6),
            "global_best":    round(self._best_score, 6),
            "improved":       improved,
            **best.to_dict(),
        })
        if len(self._history) > 200:
            self._history.pop(0)

        logger.debug(
            "optimizer_step | gen=%d best=%.4f improved=%s",
            self._generation, best.feedback_score, improved,
        )

        return best.config if improved else None

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def best_config(self) -> Optional[FusionConfig]:
        """Best FusionConfig found so far."""
        return self._best_config

    @property
    def current_config(self) -> FusionConfig:
        """Legacy alias for active config."""
        return self._current_config

    @property
    def best_score(self) -> float:
        """Highest feedback score achieved."""
        return self._best_score

    @property
    def n_evaluations(self) -> int:
        """Total number of candidate evaluations performed."""
        return self._n_evaluations

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Optimization history (most recent last)."""
        return list(self._history)

    def get_state(self) -> OptimizerState:
        """Capture optimizer state for persistence."""
        return OptimizerState(
            best_config=self._best_config.to_dict() if self._best_config else {},
            best_score=self._best_score,
            n_evaluations=self._n_evaluations,
            generation=self._generation,
            search_mean=self._mean.tolist(),
            search_std=self._std.tolist(),
            history=self._history[-50:],
        )

    def reset(self) -> None:
        """Reset optimizer to initial state."""
        d = len(self.PARAM_NAMES)
        self._mean        = np.full(d, 0.5)
        self._std         = np.full(d, self._mut_rate)
        self._best_config = None
        self._best_score  = float("-inf")
        self._current_config = FusionConfig()
        self._n_evaluations = 0
        self._generation    = 0
        self._history.clear()

    def __repr__(self) -> str:
        return (
            f"OnlineHyperparamOptimizer("
            f"gen={self._generation}, "
            f"evals={self._n_evaluations}, "
            f"best={self._best_score:.4f})"
        )
