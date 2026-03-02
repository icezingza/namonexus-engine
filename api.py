"""
NamoNexus Fusion Engine API Gateway.

Production defaults:
- Fail-closed authentication (no hardcoded API key fallback).
- Input bounds validation and metadata minimization.
- Basic security headers and optional strict CORS allowlist.
"""

from __future__ import annotations

import logging
import os
import secrets
import traceback
from typing import Annotated, Any, Dict, Optional

from cachetools import TTLCache
from fastapi import Depends, FastAPI, HTTPException, Path, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from namonexus_fusion.core.failover import RobustFusionPipeline

logger = logging.getLogger(__name__)

API_KEY_NAME = "X-API-Key"
SESSION_ID_PATTERN = r"^[A-Za-z0-9._:-]+$"
MODALITY_PATTERN = r"^[A-Za-z0-9_-]+$"

MAX_SESSION_ID_LEN = 64
MAX_MODALITY_LEN = 32
MAX_METADATA_KEYS = 32
MAX_METADATA_KEY_LEN = 64
MAX_METADATA_STR_LEN = 256

MAX_AUTH_FAILURES_PER_IP = 25
AUTH_FAILURE_WINDOW_SEC = 300

SENSITIVE_METADATA_KEYWORDS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "authorization",
    "cookie",
    "ssn",
    "credit",
    "card",
    "cvv",
    "iban",
    "account_number",
    "email",
    "phone",
)

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

app = FastAPI(
    title="NamoNexus Fusion Engine API",
    version="4.0.0",
    description="Multimodal Bayesian Fusion Engine with Golden Ratio Prior",
)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_expected_api_key() -> Optional[str]:
    configured = os.getenv("NAMONEXUS_API_KEY")
    if configured:
        return configured

    # Explicit opt-in for local development only.
    if _bool_env("NAMONEXUS_ALLOW_INSECURE_DEV_KEY", default=False):
        insecure_dev_key = os.getenv("NAMONEXUS_INSECURE_DEV_KEY")
        if insecure_dev_key:
            logger.warning(
                "Using insecure development API key via NAMONEXUS_INSECURE_DEV_KEY. "
                "Do not enable this in production."
            )
            return insecure_dev_key

    return None


def _allowed_origins() -> list[str]:
    raw = os.getenv("NAMONEXUS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


allowed_origins = _allowed_origins()
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", API_KEY_NAME],
        max_age=600,
    )


# Bounded in-memory state:
# - engines: per-session processing state.
# - auth_failures: brute-force throttle window keyed by client IP.
engines: TTLCache[str, RobustFusionPipeline] = TTLCache(maxsize=1000, ttl=3600)
auth_failures: TTLCache[str, int] = TTLCache(
    maxsize=10000,
    ttl=AUTH_FAILURE_WINDOW_SEC,
)


SessionId = Annotated[
    str,
    Field(min_length=1, max_length=MAX_SESSION_ID_LEN, pattern=SESSION_ID_PATTERN),
]
ModalityName = Annotated[
    str,
    Field(min_length=1, max_length=MAX_MODALITY_LEN, pattern=MODALITY_PATTERN),
]
ScoreValue = Annotated[float, Field(ge=0.0, le=1.0)]
ConfidenceValue = Annotated[float, Field(ge=0.0, le=1.0)]


class FusionUpdate(BaseModel):
    session_id: SessionId
    score: ScoreValue
    confidence: ConfidenceValue
    modality: ModalityName
    metadata: Optional[Dict[str, Any]] = None


class ResetRequest(BaseModel):
    session_id: SessionId


class FusionResponse(BaseModel):
    session_id: str
    fused_score: float
    risk_level: str
    uncertainty: float
    drift_alarm: bool
    compliance_gdpr: Optional[Dict[str, Any]] = None
    compliance_pdpa: Optional[Dict[str, Any]] = None


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _is_sensitive_metadata_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in SENSITIVE_METADATA_KEYWORDS)


def _sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not metadata:
        return None

    sanitized: Dict[str, Any] = {}
    for idx, (raw_key, raw_value) in enumerate(metadata.items()):
        if idx >= MAX_METADATA_KEYS:
            break

        key = str(raw_key).strip()[:MAX_METADATA_KEY_LEN]
        if not key or _is_sensitive_metadata_key(key):
            continue

        # Keep only scalar primitives to avoid leaking nested structures.
        if raw_value is None or isinstance(raw_value, (bool, int, float)):
            sanitized[key] = raw_value
        elif isinstance(raw_value, str):
            sanitized[key] = raw_value[:MAX_METADATA_STR_LEN]

    return sanitized or None


async def require_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header),
) -> None:
    expected_key = _resolve_expected_api_key()
    if not expected_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured",
        )

    client_ip = _client_ip(request)
    current_failures = auth_failures.get(client_ip, 0)
    if current_failures >= MAX_AUTH_FAILURES_PER_IP:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed authentication attempts",
        )

    if not api_key or not secrets.compare_digest(api_key, expected_key):
        auth_failures[client_ip] = current_failures + 1
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )

    # Successful auth resets the local failure counter.
    auth_failures.pop(client_ip, None)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for unhandled exceptions.
    Prevents leaking raw stack traces to the client and ensures a structured response.
    """
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return {
        "detail": "An internal server error occurred.",
        "type": "internal_error",
        "session_id": getattr(request.state, "session_id", None)
    }


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return response


@app.post("/v1/fusion/update", response_model=FusionResponse, dependencies=[Depends(require_api_key)])
async def update_session(payload: FusionUpdate):
    if payload.session_id not in engines:
        engines[payload.session_id] = RobustFusionPipeline()

    engine = engines[payload.session_id]
    request.state.session_id = payload.session_id  # For exception handler context
    safe_metadata = _sanitize_metadata(payload.metadata)

    try:
        engine.update(
            score=payload.score,
            confidence=payload.confidence,
            modality=payload.modality,
            metadata=safe_metadata,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    audit = engine.get_compliance_report()

    return FusionResponse(
        session_id=payload.session_id,
        fused_score=engine.fused_score,
        risk_level=engine.risk_level,
        uncertainty=engine.uncertainty,
        drift_alarm=engine.has_drift_alarm,
        compliance_gdpr={
            "article_22_justification": audit.get("narrative"),
            "attributions": audit.get("modality_attributions"),
            "timestamp": audit.get("timestamp"),
        },
        compliance_pdpa={
            "section_40_audit": audit,
            "processor": os.getenv("NAMONEXUS_PROCESSOR_NAME", "NamoNexus v4.0"),
            "lawful_basis": os.getenv("NAMONEXUS_LAWFUL_BASIS", "unspecified"),
        },
    )


@app.post("/v1/fusion/reset", dependencies=[Depends(require_api_key)])
async def reset_session(payload: ResetRequest):
    if payload.session_id in engines:
        engines[payload.session_id].reset_session()
        return {"status": "reset", "session_id": payload.session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@app.get("/v1/diagnostics/{session_id}", dependencies=[Depends(require_api_key)])
async def get_diagnostics(
    session_id: Annotated[
        str,
        Path(min_length=1, max_length=MAX_SESSION_ID_LEN, pattern=SESSION_ID_PATTERN),
    ],
):
    if session_id not in engines:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        return engines[session_id].get_diagnostics()
    except Exception as exc:
        logger.error("Failed to retrieve diagnostics for session %s: %s", session_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve engine diagnostics"
        ) from exc


@app.get("/v1/health")
async def health_check():
    return {
        "status": "ok",
        "active_sessions": len(engines),
        "cache_info": {"maxsize": engines.maxsize, "ttl": engines.ttl},
        "auth": {"configured": bool(_resolve_expected_api_key())},
        "cors": {"allowed_origins": len(allowed_origins)},
    }
