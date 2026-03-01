# NamoNexus Compliance Checklist (Release Gate)

This checklist is intended for commercial release readiness.
Items marked "External" require action outside the code repository.

## 1) OSS License Compliance
- Keep `LICENSE` and `NOTICE.txt` in all release bundles and container images.
- Keep dependency versions in `requirements.txt` within approved ranges.
- Re-generate SBOM before each release:
  - `python scripts/generate_sbom.py --requirements requirements.txt --output artifacts/sbom.cdx.json`
- Archive third-party license evidence from the build environment (External).

## 2) Security and Secrets
- Set `NAMONEXUS_API_KEY` in secret manager (External).
- Keep `NAMONEXUS_ALLOW_INSECURE_DEV_KEY=false` in production.
- Restrict `NAMONEXUS_ALLOWED_ORIGINS` to explicit HTTPS origins.
- Rotate API keys on incident or personnel change (External).

## 3) Privacy / PDPA / GDPR / CCPA
- Define lawful basis for processing (`NAMONEXUS_LAWFUL_BASIS`) with legal sign-off (External).
- Define retention policy for session/audit logs (External).
- Implement data-subject request process (access/deletion/export) (External).
- Confirm metadata sent to `/v1/fusion/update` does not contain direct identifiers.

## 4) IP / Patent / Trademark
- Verify contributor IP assignment and employment/contract ownership (External).
- Run trademark conflict search for product name/logo in target markets (External).
- Run prior-art and freedom-to-operate review with counsel (External).
- Ensure marketing claims do not overstate patent status (External).

## 5) Operational Readiness
- Maintain incident runbook and on-call contact list (External).
- Keep backup/restore validation evidence for deployment data stores (External).
- Document rollback procedure for API and model release (External).
