import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# MOCKING CORE DEPENDENCIES
# We mock the internal package before importing api.py to allow testing
# the API layer in isolation without the full fusion engine installed.
# ---------------------------------------------------------------------------
mock_fusion = MagicMock()
sys.modules["namonexus_fusion"] = mock_fusion
sys.modules["namonexus_fusion.core"] = mock_fusion.core
sys.modules["namonexus_fusion.core.failover"] = mock_fusion.core.failover

# Add project root to path to import api.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api import app, engines

client = TestClient(app)

@pytest.fixture
def valid_api_key(monkeypatch):
    """Sets a valid API key in the environment."""
    key = "test-secret-key-123"
    monkeypatch.setenv("NAMONEXUS_API_KEY", key)
    return key

@pytest.fixture
def mock_pipeline():
    """Patches the RobustFusionPipeline class used in api.py."""
    with patch("api.RobustFusionPipeline") as MockClass:
        instance = MockClass.return_value
        # Default mock behaviors
        instance.fused_score = 0.75
        instance.risk_level = "low"
        instance.uncertainty = 0.15
        instance.has_drift_alarm = False
        instance.get_compliance_report.return_value = {
            "narrative": "Test narrative",
            "modality_attributions": {},
            "timestamp": "2026-01-01T12:00:00Z"
        }
        instance.get_diagnostics.return_value = {"status": "healthy", "uptime": 100}
        yield instance

def test_health_check():
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "active_sessions" in data

def test_auth_missing_configuration(monkeypatch):
    """Test 503 when no API key is configured on the server."""
    monkeypatch.delenv("NAMONEXUS_API_KEY", raising=False)
    monkeypatch.delenv("NAMONEXUS_ALLOW_INSECURE_DEV_KEY", raising=False)
    
    response = client.post("/v1/fusion/update", json={})
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]

def test_auth_missing_header(valid_api_key):
    """Test 403 when X-API-Key header is missing."""
    response = client.post("/v1/fusion/update", json={})
    assert response.status_code == 403

def test_auth_invalid_key(valid_api_key):
    """Test 403 when X-API-Key is incorrect."""
    response = client.post(
        "/v1/fusion/update", 
        headers={"X-API-Key": "wrong-key"},
        json={}
    )
    assert response.status_code == 403

def test_update_session_success(valid_api_key, mock_pipeline):
    """Test successful fusion update."""
    payload = {
        "session_id": "sess_001",
        "score": 0.85,
        "confidence": 0.90,
        "modality": "text"
    }
    response = client.post(
        "/v1/fusion/update",
        headers={"X-API-Key": valid_api_key},
        json=payload
    )
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "sess_001"
    assert data["fused_score"] == 0.75  # From mock
    assert mock_pipeline.update.called

def test_update_validation_error(valid_api_key):
    """Test Pydantic validation (score > 1.0)."""
    payload = {
        "session_id": "sess_001",
        "score": 1.5,  # Invalid
        "confidence": 0.9,
        "modality": "text"
    }
    response = client.post(
        "/v1/fusion/update",
        headers={"X-API-Key": valid_api_key},
        json=payload
    )
    assert response.status_code == 422

def test_metadata_scrubbing(valid_api_key, mock_pipeline):
    """Test that sensitive keys are removed from metadata."""
    payload = {
        "session_id": "sess_scrub",
        "score": 0.5,
        "confidence": 0.5,
        "modality": "face",
        "metadata": {
            "safe_param": "ok",
            "user_password": "secret_password",
            "credit_card": "1234-5678"
        }
    }
    
    client.post(
        "/v1/fusion/update",
        headers={"X-API-Key": valid_api_key},
        json=payload
    )
    
    # Check arguments passed to engine.update
    args, kwargs = mock_pipeline.update.call_args
    metadata_arg = kwargs.get("metadata")
    
    assert "safe_param" in metadata_arg
    assert "user_password" not in metadata_arg
    assert "credit_card" not in metadata_arg

def test_reset_session(valid_api_key, mock_pipeline):
    # Inject a session into the cache
    engines["sess_reset"] = mock_pipeline
    
    response = client.post(
        "/v1/fusion/reset",
        headers={"X-API-Key": valid_api_key},
        json={"session_id": "sess_reset"}
    )
    assert response.status_code == 200
    assert mock_pipeline.reset_session.called