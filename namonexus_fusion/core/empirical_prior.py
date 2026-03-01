"""
Empirical Prior Learning — Feature 1.2
=======================================
Patent-Pending Technology | NamoNexus Research Team

Problem
-------
The base engine uses a fixed prior α₀/β₀ = φ for every person and every
context.  In practice:

  • A naturally animated speaker may always score 0.4 on voice — not
    because they are stressed, but because that is their baseline.
  • A stoic person may always score 0.7 — not because they are calm,
    but because that is their default expression.

Using a universal prior means the engine is biased toward or away from
each individual's true baseline, leading to systematically wrong risk
classifications, especially in the first few observations of a session.

Solution: Personalized Prior via Maximum Likelihood Estimation
--------------------------------------------------------------
Given a set of historical sessions H = {(s₁, c₁), ..., (sₙ, cₙ)} for
an individual or context, we estimate the Beta prior parameters (α̂, β̂)
that maximise the likelihood of having observed those sessions.

Crucially, the Golden Ratio constraint is maintained as a **soft
regularizer** — not abandoned.  The learned prior satisfies:

    α̂/β̂  ≈  φ  (when no data available)
    α̂/β̂ → empirical ratio (as data accumulates)

with a regularization strength that controls how quickly the prior
deviates from the Golden Ratio anchor.

Patent Claim (new — Claim 7)
-----------------------------
"A method of learning a personalized prior distribution for a multimodal
Bayesian fusion engine from historical session data, wherein the Golden
Ratio (φ) serves as a regularization anchor: the estimated parameters
α̂, β̂ are found by minimizing a penalized negative log-likelihood where
the penalty term is proportional to (α̂/β̂ − φ)², such that the prior
reverts to the Golden Ratio prior in the absence of data."

Synthetic Data Generator
------------------------
Since real data is not yet available, this module includes a
``SyntheticSessionGenerator`` that produces realistic multimodal session
data with configurable ground-truth states, modality noise profiles, and
inter-session variability.  This supports testing, CI, and demonstrating
the learning capability to patent examiners.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize  # type: ignore
from scipy.special import betaln      # type: ignore

from ..core.constants import GOLDEN_RATIO, GOLDEN_RATIO_RECIPROCAL
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ModalityObservation:
    """A single (score, confidence) observation from one modality."""

    score:      float
    confidence: float
    modality:   str
    timestamp:  float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0,1], got {self.score}.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0,1], got {self.confidence}.")


@dataclass
class Session:
    """
    A complete interaction session consisting of multiple observations.

    Parameters
    ----------
    session_id:
        Unique identifier.
    subject_id:
        Identifier of the individual.  Used to group sessions for
        personalized prior learning.
    observations:
        All modality observations in this session.
    ground_truth:
        Optional ground-truth label: 'calm', 'stressed', 'deceptive'.
        Used for evaluation but NOT for prior learning (unsupervised).
    """

    session_id:   str
    subject_id:   str
    observations: List[ModalityObservation]
    ground_truth: Optional[str] = None
    metadata:     Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        """Weighted mean score (weighted by confidence)."""
        if not self.observations:
            return GOLDEN_RATIO_RECIPROCAL
        weights = [o.confidence for o in self.observations]
        scores  = [o.score      for o in self.observations]
        total_w = sum(weights)
        if total_w == 0:
            return float(np.mean(scores))
        return float(sum(s * w for s, w in zip(scores, weights)) / total_w)

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Prior learning configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorLearningConfig:
    """
    Configuration for EmpiricalPriorLearner.

    Parameters
    ----------
    phi_regularization:
        Strength of the Golden Ratio regularization penalty.
        0 = no regularization (pure MLE).
        High values → prior stays close to φ anchor.
        Default = 1.0 (balanced).
    min_sessions:
        Minimum number of sessions required before the learned prior
        overrides the default Golden Ratio prior.
    min_observations_per_session:
        Sessions with fewer observations than this are excluded from
        learning (too noisy).
    update_strategy:
        'full'        — refit from scratch on every call (accurate, slow).
        'incremental' — update running sufficient statistics only (fast).
    prior_strength_bounds:
        (min, max) bounds for the total prior strength α₀ + β₀.
    """

    phi_regularization: float = 1.0
    min_sessions: int = 5
    min_observations_per_session: int = 2
    update_strategy: str = "incremental"
    prior_strength_bounds: Tuple[float, float] = (0.5, 50.0)

    def __post_init__(self) -> None:
        if self.phi_regularization < 0:
            raise ConfigurationError(
                f"phi_regularization must be >= 0, got {self.phi_regularization}."
            )
        if self.min_sessions < 1:
            raise ConfigurationError(
                f"min_sessions must be >= 1, got {self.min_sessions}."
            )
        if self.update_strategy not in ("full", "incremental"):
            raise ConfigurationError(
                f"update_strategy must be 'full' or 'incremental', "
                f"got {self.update_strategy!r}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Learned prior result
# ---------------------------------------------------------------------------


@dataclass
class LearnedPrior:
    """
    The result of fitting empirical data to a Beta prior.

    The Golden Ratio anchor is always included in the output for
    transparency and patent documentation.
    """

    alpha0: float
    beta0:  float
    phi_ratio: float                    # actual α₀/β₀
    golden_ratio_deviation: float       # |φ_ratio − φ|
    n_sessions_used: int
    n_observations_used: int
    log_likelihood: float
    subject_id: Optional[str] = None
    fitted_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )

    @property
    def prior_mean(self) -> float:
        return self.alpha0 / (self.alpha0 + self.beta0)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"LearnedPrior(α₀={self.alpha0:.4f}, β₀={self.beta0:.4f}, "
            f"mean={self.prior_mean:.4f}, φ_ratio={self.phi_ratio:.4f}, "
            f"Δφ={self.golden_ratio_deviation:.4f}, "
            f"sessions={self.n_sessions_used})"
        )


# ---------------------------------------------------------------------------
# Core learner
# ---------------------------------------------------------------------------


class EmpiricalPriorLearner:
    """
    Learns a personalized Beta prior from historical session data,
    with the Golden Ratio as a regularization anchor.

    Usage
    -----
    ::

        learner = EmpiricalPriorLearner(config=PriorLearningConfig())
        learner.add_sessions(sessions)
        prior = learner.fit(subject_id="user_001")
        # Use prior.alpha0, prior.beta0 to initialise GoldenBayesianFusion

    Parameters
    ----------
    config:
        Learning configuration.
    """

    def __init__(self, config: Optional[PriorLearningConfig] = None) -> None:
        self._config = config or PriorLearningConfig()
        self._sessions: Dict[str, List[Session]] = {}   # subject_id → sessions
        self._default_alpha0 = GOLDEN_RATIO * 2.0        # default prior strength
        self._default_beta0  = 2.0

        logger.info(
            "EmpiricalPriorLearner | φ_reg=%.2f min_sessions=%d strategy=%s",
            self._config.phi_regularization,
            self._config.min_sessions,
            self._config.update_strategy,
        )

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------

    def add_session(self, session: Session) -> None:
        """Add one session to the learning pool."""
        sid = session.subject_id
        if sid not in self._sessions:
            self._sessions[sid] = []
        self._sessions[sid].append(session)
        logger.debug(
            "Added session %s for subject %s (%d obs)",
            session.session_id, sid, session.n_observations,
        )

    def add_sessions(self, sessions: List[Session]) -> None:
        """Add multiple sessions at once."""
        for s in sessions:
            self.add_session(s)

    @property
    def subject_ids(self) -> List[str]:
        """List of all subject IDs with at least one session."""
        return list(self._sessions.keys())

    def session_count(self, subject_id: str) -> int:
        return len(self._sessions.get(subject_id, []))

    # ------------------------------------------------------------------
    # Internal: data extraction
    # ------------------------------------------------------------------

    def _extract_scores(self, subject_id: str) -> List[float]:
        """
        Extract weighted mean scores from all valid sessions of a subject.
        """
        min_obs = self._config.min_observations_per_session
        scores = []
        for s in self._sessions.get(subject_id, []):
            if s.n_observations >= min_obs:
                scores.append(s.mean_score)
        return scores

    # ------------------------------------------------------------------
    # MLE with Golden Ratio regularization
    # ------------------------------------------------------------------

    def _neg_log_likelihood(
        self,
        params: np.ndarray,
        scores: List[float],
        reg: float,
    ) -> float:
        """
        Penalized negative log-likelihood for Beta(α, β).

        NLL(α, β) = −Σᵢ log Beta_pdf(sᵢ; α, β)
                  + reg · (α/β − φ)²

        We work in log-space (params = [log α, log β]) to ensure
        positivity without explicit constraints.
        """
        log_alpha, log_beta = params
        alpha = np.exp(log_alpha)
        beta  = np.exp(log_beta)

        # Check strength bounds
        lo, hi = self._config.prior_strength_bounds
        strength = alpha + beta
        if not (lo <= strength <= hi):
            return 1e12

        # Log-likelihood: sum of log Beta PDF over scores
        # log Beta_pdf(x; α, β) = (α−1)log(x) + (β−1)log(1−x) − log B(α,β)
        log_b = betaln(alpha, beta)
        ll = sum(
            (alpha - 1) * math.log(max(s, 1e-9))
            + (beta  - 1) * math.log(max(1 - s, 1e-9))
            - log_b
            for s in scores
        )

        # Regularization: penalize deviation from Golden Ratio ratio
        ratio = alpha / beta
        penalty = reg * (ratio - GOLDEN_RATIO) ** 2

        return -ll + penalty

    def fit(self, subject_id: str) -> LearnedPrior:
        """
        Fit a personalized prior for *subject_id*.

        If insufficient data is available (< ``min_sessions``), returns
        the default Golden Ratio prior unchanged.

        Parameters
        ----------
        subject_id:
            The individual to fit a prior for.

        Returns
        -------
        LearnedPrior
            Contains alpha0, beta0, diagnostics.
        """
        scores = self._extract_scores(subject_id)
        n_sessions = self.session_count(subject_id)

        # Fall back to Golden Ratio prior if not enough data
        if n_sessions < self._config.min_sessions or len(scores) < 2:
            logger.info(
                "subject=%s: only %d sessions (need %d) — using Golden Ratio prior",
                subject_id, n_sessions, self._config.min_sessions,
            )
            return LearnedPrior(
                alpha0=self._default_alpha0,
                beta0=self._default_beta0,
                phi_ratio=GOLDEN_RATIO,
                golden_ratio_deviation=0.0,
                n_sessions_used=0,
                n_observations_used=0,
                log_likelihood=float("-inf"),
                subject_id=subject_id,
            )

        # Initialise optimisation near the Golden Ratio prior
        x0 = np.array([
            math.log(self._default_alpha0),
            math.log(self._default_beta0),
        ])

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(scores, self._config.phi_regularization),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6},
        )

        alpha_hat = float(np.exp(result.x[0]))
        beta_hat  = float(np.exp(result.x[1]))
        ratio     = alpha_hat / beta_hat

        prior = LearnedPrior(
            alpha0=alpha_hat,
            beta0=beta_hat,
            phi_ratio=ratio,
            golden_ratio_deviation=abs(ratio - GOLDEN_RATIO),
            n_sessions_used=n_sessions,
            n_observations_used=len(scores),
            log_likelihood=float(-result.fun),
            subject_id=subject_id,
        )

        logger.info("Fitted prior for subject=%s: %s", subject_id, prior)
        return prior

    def fit_all(self) -> Dict[str, LearnedPrior]:
        """Fit a prior for every known subject."""
        return {sid: self.fit(sid) for sid in self.subject_ids}

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def export_sessions(self, subject_id: str) -> str:
        """Export all sessions for a subject as JSON."""
        return json.dumps(
            [s.to_dict() for s in self._sessions.get(subject_id, [])],
            indent=2,
        )

    def __repr__(self) -> str:
        n_subjects = len(self._sessions)
        n_sessions = sum(len(v) for v in self._sessions.values())
        return (
            f"EmpiricalPriorLearner("
            f"subjects={n_subjects}, sessions={n_sessions}, "
            f"φ_reg={self._config.phi_regularization})"
        )


# ---------------------------------------------------------------------------
# Synthetic Data Generator
# ---------------------------------------------------------------------------


@dataclass
class SubjectProfile:
    """
    Defines the characteristics of a synthetic subject.

    Parameters
    ----------
    subject_id:
        Unique identifier.
    baseline_score:
        True baseline emotional score for this subject (0 = always tense,
        1 = always calm).  This is what the prior should converge to.
    voice_noise_std:
        Standard deviation of Gaussian noise added to voice scores.
    face_noise_std:
        Standard deviation of Gaussian noise added to face scores.
    text_bias:
        Systematic bias added to text scores (positive = tends to
        report better than they feel — deception tendency).
    """

    subject_id:      str
    baseline_score:  float = 0.6
    voice_noise_std: float = 0.08
    face_noise_std:  float = 0.10
    text_bias:       float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 <= self.baseline_score <= 1.0):
            raise ValueError("baseline_score must be in [0, 1].")


class SyntheticSessionGenerator:
    """
    Generates realistic synthetic multimodal session data for testing,
    benchmarking, and demonstrating Empirical Prior Learning.

    The generator models:
    - Per-subject baseline scores (to demonstrate prior personalisation)
    - Per-modality noise profiles (voice is noisier than text)
    - Random state episodes (calm, stressed, deceptive) within sessions
    - Confidence levels that vary with signal quality

    Parameters
    ----------
    seed:
        Random seed for reproducibility.

    Examples
    --------
    ::

        gen = SyntheticSessionGenerator(seed=42)

        profiles = [
            SubjectProfile("user_calm",  baseline_score=0.80),
            SubjectProfile("user_tense", baseline_score=0.35),
        ]

        sessions = gen.generate(profiles, sessions_per_subject=20)
        print(f"Generated {len(sessions)} sessions")
    """

    MODALITIES = ["text", "voice", "face"]

    # Confidence profiles per modality (mean, std) — voice is least reliable
    CONFIDENCE_PROFILES: Dict[str, Tuple[float, float]] = {
        "text":  (0.80, 0.10),
        "voice": (0.70, 0.15),
        "face":  (0.65, 0.12),
    }

    # State definitions: (score_offset, label)
    STATES: List[Tuple[float, str]] = [
        ( 0.00, "calm"),
        (-0.30, "stressed"),
        ( 0.00, "deceptive"),   # deceptive: text bias increases
    ]
    STATE_WEIGHTS = [0.50, 0.35, 0.15]   # base probabilities

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)
        logger.info("SyntheticSessionGenerator | seed=%d", seed)

    def _generate_observation(
        self,
        modality: str,
        true_score: float,
        profile: SubjectProfile,
        state_label: str,
    ) -> ModalityObservation:
        """Generate one noisy observation for a given modality."""
        noise_std = {
            "text":  0.05,
            "voice": profile.voice_noise_std,
            "face":  profile.face_noise_std,
        }[modality]

        score = true_score + self._rng.normal(0, noise_std)

        # Deceptive subjects report higher text scores
        if state_label == "deceptive" and modality == "text":
            score += profile.text_bias + self._rng.uniform(0.1, 0.25)

        score = float(np.clip(score, 0.02, 0.98))

        # Confidence varies around modality baseline
        conf_mean, conf_std = self.CONFIDENCE_PROFILES[modality]
        confidence = float(np.clip(
            self._rng.normal(conf_mean, conf_std), 0.2, 0.99
        ))

        return ModalityObservation(
            score=score,
            confidence=confidence,
            modality=modality,
        )

    def generate_session(
        self,
        profile: SubjectProfile,
        session_id: str,
        n_observations: int = 4,
    ) -> Session:
        """Generate one synthetic session for a subject."""
        # Pick a random state for this session
        state_idx = self._rng.choice(
            len(self.STATES), p=self.STATE_WEIGHTS
        )
        score_offset, state_label = self.STATES[state_idx]
        true_score = float(np.clip(
            profile.baseline_score + score_offset, 0.05, 0.95
        ))

        observations = []
        for i in range(n_observations):
            modality = self.MODALITIES[i % len(self.MODALITIES)]
            obs = self._generate_observation(
                modality, true_score, profile, state_label
            )
            observations.append(obs)

        return Session(
            session_id=session_id,
            subject_id=profile.subject_id,
            observations=observations,
            ground_truth=state_label,
            metadata={"true_score": true_score, "state": state_label},
        )

    def generate(
        self,
        profiles: List[SubjectProfile],
        sessions_per_subject: int = 20,
        obs_per_session_range: Tuple[int, int] = (3, 6),
    ) -> List[Session]:
        """
        Generate a full synthetic dataset.

        Parameters
        ----------
        profiles:
            List of subject profiles to generate data for.
        sessions_per_subject:
            Number of sessions per subject.
        obs_per_session_range:
            (min, max) observations per session.

        Returns
        -------
        List[Session]
            All sessions, shuffled randomly.
        """
        sessions: List[Session] = []

        for profile in profiles:
            for i in range(sessions_per_subject):
                n_obs = int(self._rng.integers(
                    obs_per_session_range[0],
                    obs_per_session_range[1] + 1,
                ))
                session_id = f"{profile.subject_id}_s{i:04d}"
                session = self.generate_session(profile, session_id, n_obs)
                sessions.append(session)

        self._rng.shuffle(sessions)
        logger.info(
            "Generated %d sessions for %d subjects",
            len(sessions), len(profiles),
        )
        return sessions

    def generate_standard_benchmark(self) -> Tuple[List[SubjectProfile], List[Session]]:
        """
        Generate a standard benchmark dataset with 5 subjects representing
        distinct personality/baseline profiles.

        Returns
        -------
        (profiles, sessions)
        """
        profiles = [
            SubjectProfile("subject_calm",    baseline_score=0.82, voice_noise_std=0.06),
            SubjectProfile("subject_tense",   baseline_score=0.28, voice_noise_std=0.12),
            SubjectProfile("subject_neutral", baseline_score=0.55, voice_noise_std=0.09),
            SubjectProfile("subject_deceptive", baseline_score=0.60, text_bias=0.20),
            SubjectProfile("subject_variable",  baseline_score=0.50, voice_noise_std=0.18,
                           face_noise_std=0.20),
        ]
        sessions = self.generate(profiles, sessions_per_subject=30)
        return profiles, sessions
