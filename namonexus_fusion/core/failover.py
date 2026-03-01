# © 2026 Kanin Raksaraj (P'Ice). All Rights Reserved.
# Confidential and Proprietary. Do not distribute without permission.

"""
failover.py — Robust Fusion Pipeline & Failover System (Issue 5)
================================================================
Wraps the core fusion engine to provide:
1. Resilience: Automatic filtering of low-confidence signals.
2. Failover: Handling of sensor quarantine and single-sensor fallback.
3. Continuity: Ensuring Bayesian history is preserved (no auto-clear).
"""

import logging
from typing import Any, Dict, Optional

from .phase4_fusion import Phase4GoldenFusion
from .constants import LOW_CONF_THRESHOLD

logger = logging.getLogger(__name__)

class RobustFusionPipeline:
    """
    Production-grade wrapper for the fusion engine.
    
    Implements the 'Failover' logic:
    - Filters noise (confidence < LOW_CONF_THRESHOLD).
    - Manages sensor quarantine via the underlying engine's TrustScorer.
    - Maintains Bayesian continuity (Issue 2).
    """

    def __init__(self, **kwargs: Any) -> None:
        # Initialize the full Phase 4 engine (includes TrustScorer for quarantine)
        self.engine = Phase4GoldenFusion(**kwargs)

    def update(
        self,
        score: float,
        confidence: float,
        modality: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Robust update method.
        
        Logic:
        1. Check Confidence (Issue 4): Drop if < 0.15 to prevent noise contamination.
        2. Bayesian Update (Issue 2): Accumulate evidence; DO NOT clear history.
        3. Failover (Issue 5): If sensors are quarantined, the engine automatically
           rebalances weights based on the Golden Ratio prior and remaining active sensors.
        """
        # 1. Filter low confidence signals (Issue 4)
        if confidence < LOW_CONF_THRESHOLD:
            logger.debug(
                "Signal dropped: confidence %.2f < threshold %.2f (modality=%s)",
                confidence, LOW_CONF_THRESHOLD, modality
            )
            return

        # 2. Delegate to core engine
        # The engine handles quarantine (SensorTrustScorer) and prior influence automatically.
        self.engine.update(score, confidence, modality, metadata=metadata)

    def reset_session(self) -> None:
        """
        Explicitly reset the session history.
        Operator must call this manually; it is never called by update().
        """
        self.engine.reset()
        logger.info("Session reset triggered by operator.")

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Expose internal engine state for transparency (Issue 7).
        Shows sensor calibration status (Trust Alpha/Beta) via calibration_report.
        """
        return {
            "calibration_status": self.engine.calibration_report(),
            "drift_status": self.engine.drift_summary(),
            "active_sensors": self.engine.active_modalities,
            "dropped_observations": self.engine.dropped_observations
        }

    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate XAI report for GDPR/PDPA compliance (Issue 6)."""
        report = self.engine.explain()
        return report.to_audit_dict()

    # ─── Property Proxies ─────────────────────────────────────────────────────
    
    @property
    def fused_score(self) -> float: return self.engine.fused_score
    @property
    def risk_level(self) -> str: return self.engine.risk_level
    @property
    def uncertainty(self) -> float: return self.engine.uncertainty
    @property
    def has_drift_alarm(self) -> bool: return self.engine.has_drift_alarm