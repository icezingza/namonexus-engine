"""
In-process smoke test for API endpoints without Docker/network process spawning.

Usage:
    python scripts/local_smoke_test.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure repo root is importable when running from scripts/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set defaults before importing api (module reads env at import time).
os.environ["NAMONEXUS_API_KEY"] = os.getenv("NAMONEXUS_API_KEY", "local-prod-key-123")
os.environ["NAMONEXUS_ALLOWED_ORIGINS"] = os.getenv(
    "NAMONEXUS_ALLOWED_ORIGINS", "https://app.example.com"
)
os.environ["NAMONEXUS_LAWFUL_BASIS"] = os.getenv("NAMONEXUS_LAWFUL_BASIS", "contract")

import api


def main() -> None:
    api.engines.clear()
    api.auth_failures.clear()
    client = TestClient(api.app)

    health = client.get("/v1/health")
    update = client.post(
        "/v1/fusion/update",
        headers={"X-API-Key": os.environ["NAMONEXUS_API_KEY"]},
        json={
            "session_id": "local_smoke_001",
            "score": 0.72,
            "confidence": 0.88,
            "modality": "text",
            "metadata": {
                "device": "mobile",
                "auth_token": "should_be_dropped",
                "nested": {"x": 1},
            },
        },
    )
    unauth = client.post(
        "/v1/fusion/update",
        json={
            "session_id": "local_smoke_unauth",
            "score": 0.5,
            "confidence": 0.8,
            "modality": "text",
        },
    )

    print("HEALTH_STATUS", health.status_code)
    print("HEALTH_BODY", json.dumps(health.json(), ensure_ascii=False))
    print("UPDATE_STATUS", update.status_code)
    print("UPDATE_BODY", json.dumps(update.json(), ensure_ascii=False))
    print("UNAUTH_STATUS", unauth.status_code)
    print("UNAUTH_BODY", json.dumps(unauth.json(), ensure_ascii=False))


if __name__ == "__main__":
    main()
