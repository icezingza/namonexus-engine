"""
Hierarchical Bayesian Model — Feature 4.2
==========================================
Patent-Pending Technology | NamoNexus Research Team

Problem
-------
Every NamoNexus deployment currently maintains a separate, isolated
fusion engine per individual.  This creates two concrete failures:

  1. **Cold-start problem** — for a new individual with no history, the
     engine must rely entirely on the universal Golden Ratio prior
     (alpha0/beta0 = phi) for the first several sessions, leading to
     poor calibration and high uncertainty.

  2. **Siloed learning** — useful pattern knowledge from thousands of
     sessions in Organisation A cannot improve calibration for new users
     in Organisation B, even if those users exhibit structurally similar
     behaviour profiles.

The naive solution — sharing raw observation data across organisations —
is impossible under GDPR, PDPA, HIPAA, and most enterprise data
governance policies.

Solution: Hierarchical Bayesian with Golden Ratio Structural Constraint
-----------------------------------------------------------------------
We decompose the prior into two levels:

    Level 1 (Population):   Beta(alpha_pop, beta_pop)
        Shared knowledge about the aggregate distribution of scores
        across all observed individuals within a cohort.
        Updated via federated aggregation — no raw data shared.

    Level 2 (Individual):   Beta(alpha_ind, beta_ind)
        Personalised posterior for a specific individual.
        Initialised from the population prior (warm start) and updated
        via private local observations only.

The **Golden Ratio structural constraint** is maintained at both levels:

    Population:   alpha_pop / beta_pop  initialized to  phi
    Individual:   alpha_ind / beta_ind  initialized to  phi
    Blending:     alpha_blend = rho * alpha_ind + (1-rho) * alpha_pop
                  where rho = 1 - 1/(1 + n_individual / tau)
                  and tau is the Golden Ratio-derived transition point
                  tau = phi^2 * min_individual_obs

Federated Aggregation
---------------------
Population-level updates use a secure aggregation protocol:

    1. Each site computes LOCAL sufficient statistics:
           (sum_alpha_delta, sum_beta_delta, n_sessions)
       where alpha_delta = alpha_ind - alpha0 (accumulated evidence only).

    2. The coordinator (or a secure aggregator) sums the deltas:
           alpha_pop += sum(alpha_delta_i)  /  n_sites
           beta_pop  += sum(beta_delta_i)   /  n_sites

    3. No raw scores, no individual identifiers are transmitted.

This satisfies the "minimum necessary data" principle and is compatible
with differential privacy (add Laplace noise to deltas before sharing).

Patent Claim (new — Claim 14)
------------------------------
"A hierarchical Bayesian inference architecture for multimodal fusion,
comprising:
(a) a population-level Beta prior Beta(alpha_pop, beta_pop) initialised
    with alpha_pop/beta_pop = phi (Golden Ratio), updated via federated
    aggregation of per-site sufficient statistics without raw data sharing;
(b) an individual-level posterior Beta(alpha_ind, beta_ind) initialised
    from the current population prior (warm start) and updated via private
    local observations;
(c) a blended prior alpha_blend = rho * alpha_ind + (1-rho) * alpha_pop,
    where the blending coefficient rho = 1 - 1/(1 + n / (phi^2 * tau))
    transitions from population-dominated to individual-dominated as
    n_individual observations accumulate;
(d) a federated aggregation step that sums only alpha-delta and beta-delta
    contributions from participating sites, preserving the phi structural
    constraint in the updated population prior;
such that new users benefit from warm-start calibration drawn from
population knowledge, while fully calibrated users are governed by their
private individual history, without sharing raw data across sites."
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Golden Ratio constants
_PHI: float       = (1.0 + 5.0 ** 0.5) / 2.0   # ≈ 1.618
_PHI_SQ: float    = _PHI ** 2                    # ≈ 2.618
_PHI_RECIP: float = 1.0 / _PHI                   # ≈ 0.618


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PopulationPrior:
    """
    Shared population-level Beta prior.

    Updated via federated aggregation — no raw data required.
    The phi constraint (alpha/beta = phi) is re-enforced after each
    aggregation by projecting the updated prior back onto the phi manifold.
    """

    alpha_pop:      float          # Population alpha
    beta_pop:       float          # Population beta
    n_sites:        int    = 0     # Number of sites that contributed
    n_sessions:     int    = 0     # Total sessions aggregated
    version:        int    = 0     # Incremented on each federated update
    created_at:     str    = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    updated_at:     str    = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @property
    def mean(self) -> float:
        return self.alpha_pop / (self.alpha_pop + self.beta_pop)

    @property
    def phi_ratio(self) -> float:
        return self.alpha_pop / max(self.beta_pop, 1e-12)

    @property
    def strength(self) -> float:
        return self.alpha_pop + self.beta_pop

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mean"]      = self.mean
        d["phi_ratio"] = self.phi_ratio
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class IndividualPosterior:
    """
    Private individual-level posterior.

    Initialised from the population prior (warm start) and updated via
    private local observations only.  Never shared across sites.
    """

    subject_id:     str
    alpha_ind:      float          # Individual alpha (starts = alpha_pop)
    beta_ind:       float          # Individual beta (starts = beta_pop)
    alpha0:         float          # The prior used to init this individual
    beta0:          float          # The prior used to init this individual
    n_observations: int   = 0      # Private observation count
    n_sessions:     int   = 0
    created_at:     str   = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    @property
    def mean(self) -> float:
        return self.alpha_ind / (self.alpha_ind + self.beta_ind)

    @property
    def uncertainty(self) -> float:
        a, b = self.alpha_ind, self.beta_ind
        n    = a + b
        return float(np.sqrt(a * b / (n ** 2 * (n + 1)))) if n > 2 else 0.5

    @property
    def accumulated_evidence(self) -> float:
        return max(0.0, (self.alpha_ind + self.beta_ind) - (self.alpha0 + self.beta0))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mean"]        = self.mean
        d["uncertainty"] = self.uncertainty
        return d


@dataclass
class FederatedDelta:
    """
    Sufficient statistics exported by one site for federated aggregation.

    Only deltas (not absolute values) are shared, ensuring no individual
    information is revealed.

    Fields
    ------
    site_id:
        Unique identifier for this site/organisation.
    alpha_delta:
        Sum of (alpha_ind - alpha0) across all subjects at this site.
        Represents accumulated evidence, NOT the prior itself.
    beta_delta:
        Sum of (beta_ind - beta0) across all subjects at this site.
    n_subjects:
        Number of subjects contributing deltas (for averaging).
    n_sessions:
        Total sessions across all contributing subjects.
    noise_applied:
        True if differential privacy noise was added to the deltas.
    noise_scale:
        Laplace noise scale used (epsilon for differential privacy).
    """

    site_id:       str
    alpha_delta:   float
    beta_delta:    float
    n_subjects:    int
    n_sessions:    int
    noise_applied: bool  = False
    noise_scale:   float = 0.0
    created_at:    str   = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "FederatedDelta":
        d = json.loads(s)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class BlendedPrior:
    """
    Result of blending population and individual priors.

    rho controls the individual weight:
        rho = 1 - 1 / (1 + n_ind / (phi^2 * tau))

    At n_ind = 0: rho = 0 (pure population prior)
    As n_ind → ∞: rho → 1 (pure individual posterior)
    At n_ind = phi^2 * tau: rho = 0.5 (equal blend)
    """

    alpha_blend: float
    beta_blend:  float
    rho:         float    # Individual weight in [0, 1]
    n_individual: int
    source:      str      # "population", "individual", or "blended"

    @property
    def mean(self) -> float:
        return self.alpha_blend / (self.alpha_blend + self.beta_blend)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HierarchicalConfig:
    """
    Configuration for the HierarchicalBayesianModel.

    Parameters
    ----------
    population_prior_strength:
        Initial strength (alpha + beta) of the population prior.
        alpha_pop = phi * strength, beta_pop = strength.
        Default = phi^2 ≈ 2.618 (moderate prior mass).
    individual_prior_strength:
        Overrides population prior strength for individual initialisation.
        None = use population_prior_strength.
    tau:
        Transition rate for the blending coefficient rho.
        At n_individual = phi^2 * tau observations, rho = 0.5.
        Default = 5 (transition at ~13 observations).
    min_sessions_for_blend:
        Minimum sessions before individual posterior is blended with
        population (avoids premature drift from population).
    enforce_phi_constraint:
        If True, re-project population prior after each federated update
        to maintain alpha_pop / beta_pop = phi within tolerance.
    phi_tolerance:
        Maximum allowed deviation of phi_ratio from phi.
        If exceeded and enforce_phi_constraint=True, prior is projected.
    dp_noise_scale:
        Laplace noise scale for differential privacy (0 = disabled).
        Noise is added to alpha_delta and beta_delta before export.
    max_individual_strength:
        Cap on alpha_ind + beta_ind to prevent overfitting.
    """

    population_prior_strength:  float        = _PHI_SQ       # phi^2 ≈ 2.618
    individual_prior_strength:  Optional[float] = None
    tau:                        float        = _PHI_SQ        # phi^2 ≈ 2.618
    min_sessions_for_blend:     int          = 2
    enforce_phi_constraint:     bool         = True
    phi_tolerance:              float        = 0.1
    dp_noise_scale:             float        = 0.0
    max_individual_strength:    float        = 500.0

    def __post_init__(self) -> None:
        if self.population_prior_strength <= 0:
            raise ValueError("population_prior_strength must be > 0")
        if self.tau <= 0:
            raise ValueError("tau must be > 0")
        if self.phi_tolerance < 0:
            raise ValueError("phi_tolerance must be >= 0")

    @classmethod
    def golden(cls) -> "HierarchicalConfig":
        """Default config: phi^2 population strength, tau=phi^2."""
        return cls()

    @classmethod
    def federated(cls, dp_noise: float = 0.1) -> "HierarchicalConfig":
        """Config for federated deployments with differential privacy."""
        return cls(dp_noise_scale=dp_noise)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core Hierarchical Bayesian Model
# ---------------------------------------------------------------------------


class HierarchicalBayesianModel:
    """
    Two-level Bayesian model: population prior + individual posterior.

    Manages a population-level prior shared across all subjects and
    a private individual posterior per subject.  Supports federated
    aggregation with optional differential privacy.

    Thread Safety
    -------------
    All public methods are thread-safe.  Individual posteriors and the
    population prior are protected by per-subject locks and a global lock
    respectively.

    Parameters
    ----------
    config:
        HierarchicalConfig.
    site_id:
        Identifier for this deployment site (used in FederatedDelta).

    Examples
    --------
    ::

        model = HierarchicalBayesianModel()

        # New user — warm-start from population prior
        prior = model.get_blended_prior("user_001")
        engine = Phase4GoldenFusion(hierarchical_prior=prior)

        # After a session
        model.update_individual("user_001", alpha_delta=0.5, beta_delta=0.2, n_obs=3)

        # Export federated delta (no raw data)
        delta = model.export_federated_delta()

        # Coordinator aggregates deltas from multiple sites
        model.aggregate_federated_deltas([delta_site_a, delta_site_b])

        # Next user at same site gets population-informed warm start
        prior2 = model.get_blended_prior("user_002")
    """

    def __init__(
        self,
        config:  Optional[HierarchicalConfig] = None,
        site_id: Optional[str]                = None,
    ) -> None:
        self._cfg     = config or HierarchicalConfig.golden()
        self._site_id = site_id or str(uuid.uuid4())[:8]
        self._lock    = threading.RLock()

        # Initialise population prior with Golden Ratio constraint
        s          = self._cfg.population_prior_strength
        alpha_pop0 = _PHI * s
        beta_pop0  = s
        self._population = PopulationPrior(
            alpha_pop = alpha_pop0,
            beta_pop  = beta_pop0,
        )

        self._individuals: Dict[str, IndividualPosterior] = {}
        self._ind_locks:   Dict[str, threading.RLock]     = {}

        logger.info(
            "HierarchicalBayesianModel | site=%s alpha_pop=%.4f beta_pop=%.4f phi_ratio=%.4f",
            self._site_id, alpha_pop0, beta_pop0, alpha_pop0 / beta_pop0,
        )

    # ------------------------------------------------------------------
    # Individual management
    # ------------------------------------------------------------------

    def _get_ind_lock(self, subject_id: str) -> threading.RLock:
        with self._lock:
            if subject_id not in self._ind_locks:
                self._ind_locks[subject_id] = threading.RLock()
            return self._ind_locks[subject_id]

    def register_subject(self, subject_id: str) -> IndividualPosterior:
        """
        Register a new subject and initialise their posterior from the
        current population prior (warm start).

        If the subject is already registered, return their existing posterior.

        Parameters
        ----------
        subject_id:
            Unique identifier for the subject.

        Returns
        -------
        IndividualPosterior
        """
        lock = self._get_ind_lock(subject_id)
        with lock:
            if subject_id in self._individuals:
                return self._individuals[subject_id]

            with self._lock:
                alpha0 = self._population.alpha_pop
                beta0  = self._population.beta_pop

            # Override individual prior strength if configured
            if self._cfg.individual_prior_strength is not None:
                s      = self._cfg.individual_prior_strength
                alpha0 = _PHI * s
                beta0  = s

            posterior = IndividualPosterior(
                subject_id    = subject_id,
                alpha_ind     = alpha0,
                beta_ind      = beta0,
                alpha0        = alpha0,
                beta0         = beta0,
            )
            self._individuals[subject_id] = posterior

            logger.info(
                "Registered subject=%s | warm-start alpha=%.4f beta=%.4f",
                subject_id, alpha0, beta0,
            )
            return posterior

    def update_individual(
        self,
        subject_id:  str,
        alpha_delta: float,
        beta_delta:  float,
        n_obs:       int   = 1,
        n_sessions:  int   = 1,
    ) -> IndividualPosterior:
        """
        Apply accumulated evidence from one session to an individual's posterior.

        Parameters
        ----------
        subject_id:
            Subject identifier.
        alpha_delta:
            Successes (alpha evidence) accumulated in this session.
        beta_delta:
            Failures (beta evidence) accumulated in this session.
        n_obs:
            Number of observations in this session.
        n_sessions:
            Number of sessions (usually 1).

        Returns
        -------
        IndividualPosterior
            Updated posterior.
        """
        if subject_id not in self._individuals:
            self.register_subject(subject_id)

        lock = self._get_ind_lock(subject_id)
        with lock:
            ind = self._individuals[subject_id]

            # Cap individual strength
            new_alpha = ind.alpha_ind + alpha_delta
            new_beta  = ind.beta_ind  + beta_delta
            if new_alpha + new_beta > self._cfg.max_individual_strength:
                # Scale down to max but preserve ratio
                ratio = new_alpha / max(new_alpha + new_beta, 1e-12)
                new_alpha = ratio * self._cfg.max_individual_strength
                new_beta  = (1.0 - ratio) * self._cfg.max_individual_strength

            ind.alpha_ind      = new_alpha
            ind.beta_ind       = new_beta
            ind.n_observations += n_obs
            ind.n_sessions     += n_sessions

        logger.debug(
            "update_individual | subject=%s alpha=%.4f beta=%.4f n_obs=%d",
            subject_id, new_alpha, new_beta, ind.n_observations,
        )
        return ind

    # ------------------------------------------------------------------
    # Blended prior computation
    # ------------------------------------------------------------------

    def _compute_rho(self, n_individual: int, n_sessions: int) -> float:
        """
        Compute individual weight rho in [0, 1].

        rho = 0 → pure population prior (cold start)
        rho = 1 → pure individual posterior (well-observed)
        rho = 0.5 at n = phi^2 * tau observations
        """
        if n_sessions < self._cfg.min_sessions_for_blend:
            return 0.0
        tau_obs = _PHI_SQ * self._cfg.tau
        rho     = 1.0 - 1.0 / (1.0 + n_individual / max(tau_obs, 1e-9))
        return float(np.clip(rho, 0.0, 1.0))

    def get_blended_prior(self, subject_id: str) -> BlendedPrior:
        """
        Compute the blended prior for a subject.

        For new subjects (n=0), returns the population prior directly.
        For well-observed subjects, transitions toward the individual posterior.

        Parameters
        ----------
        subject_id:
            Subject identifier.  Registers the subject if unknown.

        Returns
        -------
        BlendedPrior
        """
        if subject_id not in self._individuals:
            self.register_subject(subject_id)

        lock = self._get_ind_lock(subject_id)
        with lock:
            ind = self._individuals[subject_id]
            n_ind = ind.n_observations

        with self._lock:
            pop = self._population

        rho = self._compute_rho(n_ind, ind.n_sessions)

        alpha_blend = rho * ind.alpha_ind + (1.0 - rho) * pop.alpha_pop
        beta_blend  = rho * ind.beta_ind  + (1.0 - rho) * pop.beta_pop

        if rho == 0.0:
            source = "population"
        elif rho >= 0.95:
            source = "individual"
        else:
            source = "blended"

        logger.debug(
            "get_blended_prior | subject=%s rho=%.4f alpha=%.4f beta=%.4f source=%s",
            subject_id, rho, alpha_blend, beta_blend, source,
        )

        return BlendedPrior(
            alpha_blend  = float(alpha_blend),
            beta_blend   = float(beta_blend),
            rho          = float(rho),
            n_individual = n_ind,
            source       = source,
        )

    # ------------------------------------------------------------------
    # Federated export / aggregation
    # ------------------------------------------------------------------

    def export_federated_delta(
        self,
        subject_ids: Optional[List[str]] = None,
    ) -> FederatedDelta:
        """
        Compute and export federated sufficient statistics for this site.

        Only accumulated evidence deltas are exported — never absolute
        posteriors or raw observations.

        Parameters
        ----------
        subject_ids:
            List of subjects to include.  None = all registered subjects.

        Returns
        -------
        FederatedDelta
            Ready to send to the federated coordinator.
        """
        if subject_ids is None:
            with self._lock:
                subject_ids = list(self._individuals.keys())

        alpha_delta_sum = 0.0
        beta_delta_sum  = 0.0
        n_subjects      = 0
        n_sessions      = 0

        for sid in subject_ids:
            if sid not in self._individuals:
                continue
            lock = self._get_ind_lock(sid)
            with lock:
                ind = self._individuals[sid]
                # Delta = accumulated evidence only (prior excluded)
                alpha_delta_sum += ind.alpha_ind - ind.alpha0
                beta_delta_sum  += ind.beta_ind  - ind.beta0
                n_subjects      += 1
                n_sessions      += ind.n_sessions

        noise_applied = False
        noise_scale   = 0.0

        # Optional differential privacy: add Laplace noise
        if self._cfg.dp_noise_scale > 0:
            rng = np.random.default_rng()
            alpha_delta_sum += rng.laplace(0, self._cfg.dp_noise_scale)
            beta_delta_sum  += rng.laplace(0, self._cfg.dp_noise_scale)
            noise_applied    = True
            noise_scale      = self._cfg.dp_noise_scale

        delta = FederatedDelta(
            site_id       = self._site_id,
            alpha_delta   = float(alpha_delta_sum),
            beta_delta    = float(beta_delta_sum),
            n_subjects    = n_subjects,
            n_sessions    = n_sessions,
            noise_applied = noise_applied,
            noise_scale   = noise_scale,
        )

        logger.info(
            "Exported FederatedDelta | site=%s subjects=%d alpha_delta=%.4f beta_delta=%.4f",
            self._site_id, n_subjects, alpha_delta_sum, beta_delta_sum,
        )
        return delta

    def aggregate_federated_deltas(
        self,
        deltas: List[FederatedDelta],
        learning_rate: float = 1.0,
    ) -> PopulationPrior:
        """
        Aggregate FederatedDeltas from multiple sites into the population prior.

        The population prior is updated by averaging the per-site mean deltas
        (normalised by n_subjects) and adding to the current population prior.

        Parameters
        ----------
        deltas:
            FederatedDelta objects from all participating sites.
        learning_rate:
            Scales the update step (0 < lr <= 1).  Default = 1.0 (full update).

        Returns
        -------
        PopulationPrior
            Updated population prior.
        """
        if not deltas:
            return self._population

        total_subjects = sum(d.n_subjects for d in deltas)
        if total_subjects == 0:
            return self._population

        # Weighted average of per-subject mean deltas
        alpha_update = sum(
            d.alpha_delta / max(d.n_subjects, 1) for d in deltas
        ) / len(deltas)
        beta_update  = sum(
            d.beta_delta  / max(d.n_subjects, 1) for d in deltas
        ) / len(deltas)

        with self._lock:
            self._population.alpha_pop += learning_rate * alpha_update
            self._population.beta_pop  += learning_rate * beta_update

            # Enforce non-negativity
            self._population.alpha_pop = max(self._population.alpha_pop, _PHI * 0.1)
            self._population.beta_pop  = max(self._population.beta_pop,  0.1)

            # Re-enforce Golden Ratio constraint if configured
            if self._cfg.enforce_phi_constraint:
                phi_ratio = self._population.alpha_pop / self._population.beta_pop
                if abs(phi_ratio - _PHI) > self._cfg.phi_tolerance:
                    # Project: keep strength, restore phi ratio
                    strength = self._population.alpha_pop + self._population.beta_pop
                    self._population.alpha_pop = _PHI / (1.0 + _PHI) * strength
                    self._population.beta_pop  = 1.0  / (1.0 + _PHI) * strength
                    logger.info(
                        "Population prior phi constraint restored | phi_ratio=%.4f → %.4f",
                        phi_ratio,
                        self._population.alpha_pop / self._population.beta_pop,
                    )

            self._population.n_sites   += len(deltas)
            self._population.n_sessions += sum(d.n_sessions for d in deltas)
            self._population.version   += 1
            self._population.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        logger.info(
            "Aggregated %d federated deltas | pop alpha=%.4f beta=%.4f phi=%.4f",
            len(deltas),
            self._population.alpha_pop,
            self._population.beta_pop,
            self._population.phi_ratio,
        )
        return self._population

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def population_prior(self) -> PopulationPrior:
        with self._lock:
            return self._population

    def individual_posterior(self, subject_id: str) -> Optional[IndividualPosterior]:
        return self._individuals.get(subject_id)

    def all_subjects(self) -> List[str]:
        return list(self._individuals.keys())

    @property
    def subject_count(self) -> int:
        return len(self._individuals)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            pop = self._population.to_dict()
        return {
            "site_id":       self._site_id,
            "population":    pop,
            "n_subjects":    self.subject_count,
            "config":        self._cfg.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"HierarchicalBayesianModel("
            f"site={self._site_id}, "
            f"subjects={self.subject_count}, "
            f"pop_mean={self._population.mean:.4f}, "
            f"phi_ratio={self._population.phi_ratio:.4f})"
        )
