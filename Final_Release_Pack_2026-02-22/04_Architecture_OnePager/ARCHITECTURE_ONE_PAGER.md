# NamoNexus Fusion Engine v4.0.0 - Architecture One-Pager (Updated)

- Date: 2026-02-22
- Delivery scope: Production-ready hardened release package

## 1) System Purpose

`NamoNexus Fusion Engine` fuses multimodal signals (`score`, `confidence`, `modality`) into a Bayesian posterior risk estimate with explainability and audit outputs for compliance workflows.

## 2) High-Level Architecture

```text
Client / Upstream System
        |
        v
  FastAPI Gateway (api.py)
  - API key auth (X-API-Key)
  - Input validation + metadata sanitization
  - Session isolation (TTL cache)
        |
        v
  RobustFusionPipeline (core/failover.py)
        |
        v
  Phase4GoldenFusion (core/phase4_fusion.py)
    |- Phase3: drift detection + streaming
    |- Phase2: calibration + trust scoring + hyperopt
    |- Phase1: temporal forgetting + empirical prior
    |- Base : golden_bayesian posterior engine
        |
        v
  Explainability Layer (core/explainability.py)
  - Shapley-style modality attribution
  - GDPR/PDPA audit narrative
```

## 3) Core Runtime Flow

1. API receives update request at `/v1/fusion/update`.
2. Request is authenticated with env-based API key.
3. Payload is schema-validated and metadata is filtered for sensitive keys.
4. Engine updates posterior state and reliability signals.
5. Explainability report is generated and returned with risk + compliance payload.

## 4) Security & Hardening Controls

- Fail-closed auth configuration (no hardcoded production key fallback).
- Constant-time API key comparison.
- Brute-force throttle for repeated auth failures.
- Strict input constraints for `session_id`, `modality`, `score`, `confidence`.
- Metadata minimization to reduce accidental PII/secret propagation.
- Optional strict CORS allowlist via `NAMONEXUS_ALLOWED_ORIGINS`.
- Container runs as non-root with reduced privilege.
- Production compose override sets read-only filesystem, dropped capabilities, and `no-new-privileges`.

## 5) Deployment Topology

- Base compose: `docker-compose.yml`
- Dev override: `docker-compose.dev.yml` (hot reload + source mount)
- Prod override: `docker-compose.prod.yml` (hardened runtime settings)
- Smoke test automation:
  - `scripts/compose_smoke.ps1` (compose mode)
  - `scripts/docker_smoke.ps1` (single container mode)
  - `scripts/local_smoke_test.py` (no-docker fallback)

## 6) Quality & Release Evidence in This Pack

- Source code snapshot (hardened): `01_Source_Code_Hardened/`
- Test evidence (`216 passed`): `02_Test_Reports/`
- SBOM + compliance checklist: `03_SBOM_Compliance/`
- This architecture document: `04_Architecture_OnePager/`

## 7) Known External Dependencies (Out of Repo Scope)

- Secret manager provisioning and key rotation policy.
- Legal review for PDPA/GDPR/CCPA basis and retention policy.
- Trademark and patent/FTO searches in target jurisdictions.
