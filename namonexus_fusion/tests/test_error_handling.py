import pytest
from namonexus_fusion import NamoNexusEngine
from namonexus_fusion.core.exceptions import InvalidObservationError, MathematicalError
from namonexus_fusion.config.settings import FusionConfig
from fastapi.testclient import TestClient
from api import app
import os

def test_mathematical_edge_cases():
    # Use config object instead of keywords if the inheritance chain is sensitive
    config = FusionConfig(prior_strength=0.01)
    engine = NamoNexusEngine(config=config)
    
    # Force alpha/beta to zero 
    engine._alpha = 0.0
    engine._beta = 0.0
    
    # Verify fused_score defaults to 0.5
    assert engine.fused_score == 0.5
    # Verify uncertainty defaults to 0.5
    assert engine.uncertainty == 0.5
    # Verify interval fails safely
    lo, hi = engine.credible_interval()
    assert lo == 0.0
    assert hi == 1.0

def test_batch_update_resilience():
    engine = NamoNexusEngine()
    observations = [
        {"score": 0.8, "confidence": 0.9, "modality": "text"},
        {"score": 1.5, "confidence": 0.9, "modality": "voice"}, # Invalid score
        {"score": 0.6, "confidence": 0.85, "modality": "face"},
    ]
    
    engine.update_batch(observations)
    
    # Verify that valid observations were still processed
    # total_observations should be > 0
    assert engine.total_observations > 0
    assert len(engine.history) == 2 # Only valid ones in history

def test_api_unhandled_exception():
    client = TestClient(app)
    # Set a dummy API key
    os.environ["NAMONEXUS_API_KEY"] = "test-key"
    
    # Trigger a 422 to ensure standard FastAPI error handling still works
    response = client.post("/v1/fusion/update", 
                           json={"session_id": "s1", "score": "not-a-float", "confidence": 0.5, "modality": "text"},
                           headers={"X-API-Key": "test-key"})
    assert response.status_code == 422
    
    # Test a session not found
    response = client.get("/v1/diagnostics/non-existent", headers={"X-API-Key": "test-key"})
    assert response.status_code == 404

def test_api_key_fail_closed():
    client = TestClient(app)
    # Ensure no key in env
    if "NAMONEXUS_API_KEY" in os.environ:
        del os.environ["NAMONEXUS_API_KEY"]
    
    response = client.get("/v1/health")
    assert response.status_code == 200 # Health is public
    
    response = client.post("/v1/fusion/reset", json={"session_id": "any"}, headers={"X-API-Key": "any"})
    # Fail-closed: if not configured, returns 503 instead of allowing anything
    assert response.status_code == 503
