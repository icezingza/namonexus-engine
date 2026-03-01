# © 2026 Kanin Raksaraj (P'Ice). All Rights Reserved.
# Confidential and Proprietary. Do not distribute without permission.

from .golden_bayesian import GoldenBayesianFusion, FusionState
from .temporal_golden_fusion import TemporalGoldenFusion
from .phase2_fusion import Phase2GoldenFusion
from .phase3_fusion import Phase3GoldenFusion
from .phase4_fusion import Phase4GoldenFusion
from .fusion_engine import NamoNexusEngine

# Feature 4.1: Explainability
from .explainability import (
    ShapleyExplainer,
    ExplanationConfig,
    ExplanationReport,
    # Backward compatibility aliases
    ExplainabilityLayer,
    XAIConfig,
)

# Feature 4.2: Hierarchical Bayes
from .hierarchical_bayes import (
    PopulationModel,
    LocalModel,
    FederatedAggregator,
    HierarchicalConfig,
)
from .hierarchical_bayesian import (
    HierarchicalBayesianModel,
    PopulationPrior,
    IndividualPosterior,
    BlendedPrior,
    FederatedDelta,
)

# Feature 3.1 & 3.2: Drift & Streaming
from .drift_detector import DriftDetector, DriftConfig
from .streaming_pipeline import (
    StreamingPipeline,
    StreamingConfig,
    StreamingObservation,
    StreamResult,
    InMemoryConnector,
    KafkaConnector,
    WebSocketConnector,
)

# Constants
from .constants import GOLDEN_RATIO, GOLDEN_RATIO_RECIPROCAL, ENGINE_VERSION
