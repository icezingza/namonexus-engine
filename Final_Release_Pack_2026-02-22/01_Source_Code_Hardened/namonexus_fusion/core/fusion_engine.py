# © 2026 Kanin Raksaraj (P'Ice). All Rights Reserved.
# Confidential and Proprietary. Do not distribute without permission.

"""
fusion_engine.py — Main Entry Point & Session Management
========================================================
High-level wrapper for the NamoNexus Fusion Engine.
"""

import hashlib
import time
import logging
from typing import Any, Dict, List, Optional

from .failover import RobustFusionPipeline

logger = logging.getLogger(__name__)

class NamoNexusEngine:
    """
    The primary facade for the NamoNexus Fusion Engine.
    Combines Phase 4 capabilities with session management and PII protection.
    """

    def __init__(
        self,
        subject_id: Optional[str] = None,
        hierarchical_model: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        # Support CI usage where hierarchical_model is passed
        if hierarchical_model:
            # Extract population from wrapper if needed, or pass as is
            pop = getattr(hierarchical_model, "_population", hierarchical_model)
            kwargs['population'] = pop
        
        self.pipeline = RobustFusionPipeline(**kwargs)
        self.subject_id = subject_id
        self.session_id = self._generate_session_id(subject_id) if subject_id else None

    def _generate_session_id(self, subject_id: str) -> str:
        """
        Generate a secure session identifier.
        
        Security Fix (C2):
        Never use raw subject_id in session strings to prevent PII leaks.
        Use a one-way hash instead.
        """
        # Fix C2: Use SHA256 hash of subject_id to anonymize PII
        subject_hash = hashlib.sha256(subject_id.encode('utf-8')).hexdigest()[:16]
        timestamp = int(time.time())
        return f"{subject_hash}-{timestamp}"

    def update(self, score: float, confidence: float, modality_name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Delegate update to the robust pipeline (handles quarantine/failover)."""
        self.pipeline.update(
            score=score,
            confidence=confidence,
            modality=modality_name,
            metadata=metadata
        )

    def update_batch(self, observations: List[Dict[str, Any]]) -> None:
        """Batch update helper for CI/testing."""
        for obs in observations:
            self.update(
                score=obs["score"],
                confidence=obs["confidence"],
                modality_name=obs["modality"]
            )

    def session_summary(self) -> Dict[str, Any]:
        """Return summary of the current session state."""
        return {
            "session_id": self.session_id,
            "fused_score": self.pipeline.fused_score,
            "risk_level": self.pipeline.risk_level,
            "uncertainty": self.pipeline.uncertainty,
            "drift_alarm": self.pipeline.has_drift_alarm
        }

    def commit_session(self) -> None:
        """Commit session data (stub for CI)."""
        logger.info("Session %s committed", self.session_id)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes (e.g. explain, _alpha0) to the underlying engine."""
        return getattr(self.pipeline.engine, name)