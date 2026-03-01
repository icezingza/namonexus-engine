from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture(autouse=True)
def _reset_api_state(monkeypatch):
    api.engines.clear()
    api.auth_failures.clear()
    monkeypatch.delenv("NAMONEXUS_API_KEY", raising=False)
    monkeypatch.delenv("NAMONEXUS_ALLOW_INSECURE_DEV_KEY", raising=False)
    monkeypatch.delenv("NAMONEXUS_INSECURE_DEV_KEY", raising=False)


@pytest.fixture
def client():
    return TestClient(api.app)


def _valid_payload() -> dict:
    return {
        "session_id": "sess_001",
        "score": 0.6,
        "confidence": 0.8,
        "modality": "text",
        "metadata": {"device": "mobile"},
    }


def test_update_requires_auth_configuration(client):
    resp = client.post("/v1/fusion/update", json=_valid_payload())
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]


def test_update_rejects_missing_api_key(monkeypatch, client):
    monkeypatch.setenv("NAMONEXUS_API_KEY", "test-key")
    resp = client.post("/v1/fusion/update", json=_valid_payload())
    assert resp.status_code == 403


def test_update_accepts_valid_api_key(monkeypatch, client):
    monkeypatch.setenv("NAMONEXUS_API_KEY", "test-key")
    resp = client.post(
        "/v1/fusion/update",
        json=_valid_payload(),
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == "sess_001"
    assert 0.0 <= body["fused_score"] <= 1.0


def test_input_bounds_validation(monkeypatch, client):
    monkeypatch.setenv("NAMONEXUS_API_KEY", "test-key")
    bad = _valid_payload()
    bad["score"] = 1.5
    resp = client.post("/v1/fusion/update", json=bad, headers={"X-API-Key": "test-key"})
    assert resp.status_code == 422


def test_auth_throttle_after_repeated_failures(monkeypatch, client):
    monkeypatch.setenv("NAMONEXUS_API_KEY", "test-key")
    payload = _valid_payload()

    for _ in range(api.MAX_AUTH_FAILURES_PER_IP):
        resp = client.post(
            "/v1/fusion/update",
            json=payload,
            headers={"X-API-Key": "wrong"},
        )
        assert resp.status_code == 403

    blocked = client.post(
        "/v1/fusion/update",
        json=payload,
        headers={"X-API-Key": "wrong"},
    )
    assert blocked.status_code == 429


def test_metadata_sanitizer_drops_sensitive_and_nested():
    clean = api._sanitize_metadata(
        {
            "session_label": "lab_a",
            "auth_token": "should_drop",
            "device": "mobile",
            "nested": {"x": 1},
            "count": 7,
        }
    )
    assert clean == {"session_label": "lab_a", "device": "mobile", "count": 7}
