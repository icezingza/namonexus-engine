# NamoNexus Fusion Engine

> Patent Pending | Proprietary & Confidential | Copyright (c) 2026 NamoNexus Research Team

NamoNexus Fusion Engine is a production-focused multimodal Bayesian fusion stack. It uses the Golden Ratio (`phi`) as a structural mathematical anchor across prior initialization, temporal adaptation, trust scoring, drift detection, explainability, and hierarchical federated learning.

## English Core, Thai Soul

This repository follows an **English Core, Thai Soul** language policy:

- Code, docs, README, and reports are written in English as the primary language.
- Brand terms remain in original form:
  - `NaMo`
  - `Dhammic Moat`
  - `Karuna Protocol`

Brand framing:

- `NaMo`: the identity of the core reasoning and fusion engine.
- `Dhammic Moat`: defensibility through rigorous structure, transparency, and ethical constraints.
- `Karuna Protocol`: privacy-aware and human-centered operational discipline.

## Capability Map

| Phase | Claims | Core Capability |
|---|---:|---|
| Base (v3) | 1-5 | Beta posterior fusion, credible intervals, risk and deception metrics |
| Phase 1 | 6-7 | Temporal forgetting and empirical prior personalization |
| Phase 2 | 8-10 | Modality auto-calibration, sensor trust, online hyperparameter optimization |
| Phase 3 | 11-12 | Drift detection and streaming with delivery controls |
| Phase 4 | 13-14 | Explainability (Shapley-style) and hierarchical federated Bayesian learning |

## Installation

```bash
git clone https://github.com/namonexus/namonexus-fusion.git
cd namonexus-fusion

# development
pip install -e ".[dev]"

# production
pip install .
```

Requirements: Python 3.9+.

## Quick Start

```python
from namonexus_fusion import NamoNexusEngine

engine = NamoNexusEngine()
engine.update(0.85, 0.70, "text")
engine.update(0.25, 0.90, "voice")
engine.update(0.60, 0.85, "face")

print(engine.fused_score)
print(engine.risk_level)
print(engine.credible_interval())
```

## API Security Configuration

Set runtime environment variables before deploying `api.py`:

```bash
export NAMONEXUS_API_KEY="<strong-random-key>"
export NAMONEXUS_ALLOWED_ORIGINS="https://app.example.com"
export NAMONEXUS_LAWFUL_BASIS="contract"
```

Security notes:

- `NAMONEXUS_API_KEY` is mandatory in production.
- Keep `NAMONEXUS_ALLOW_INSECURE_DEV_KEY=false` outside local development.
- Do not send sensitive identifiers or secrets inside `metadata`.

## Testing

```bash
# full suite
PYTHONPATH=. pytest -q namonexus_fusion/tests

# selected
PYTHONPATH=. pytest -q namonexus_fusion/tests/test_api_security.py
```

Current validated status in this workspace: `216 passed`.

## SBOM and Compliance

Generate SBOM:

```bash
python scripts/generate_sbom.py --requirements requirements.txt --output artifacts/sbom.cdx.json
```

Compliance checklist:

- `docs/COMPLIANCE_CHECKLIST.md`

## Deployment

### Docker (single container smoke)

```powershell
./scripts/docker_smoke.ps1 -ApiKey "<strong-random-key>"
```

### Docker Compose (override model)

One-time env setup:

```powershell
Copy-Item .env.compose.prod.example .env.compose.prod
Copy-Item .env.compose.dev.example .env.compose.dev
```

Production mode:

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build api
```

Development mode:

```powershell
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build api
```

Compose smoke test:

```powershell
./scripts/compose_smoke.ps1 -Mode prod
./scripts/compose_smoke.ps1 -Mode dev
```

No-Docker fallback:

```bash
python scripts/local_smoke_test.py
```

## Architecture

Main package layout:

```text
namonexus_fusion/
  engine.py
  config/settings.py
  core/
    golden_bayesian.py
    temporal_golden_fusion.py
    phase2_fusion.py
    phase3_fusion.py
    phase4_fusion.py
    explainability.py
    hierarchical_bayesian.py
  tests/
```

## License

This software is proprietary.

- Terms: `LICENSE`
- Third-party attributions: `NOTICE.txt`

