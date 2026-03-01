# The Final Release Pack

This folder is the delivery bundle for production handoff.

## Language Standard

- Primary language: English
- Preserved branding terms: `NaMo`, `Dhammic Moat`, `Karuna Protocol`

## Contents

1. `01_Source_Code_Hardened/`
   - Hardened API/runtime source snapshot
   - Docker and Compose deployment definitions
   - CI workflow and release scripts

2. `02_Test_Reports/`
   - `pytest-output.txt`
   - `pytest-junit.xml`
   - `TEST_REPORT_SUMMARY.md` (`216 passed`)

3. `03_SBOM_Compliance/`
   - `sbom.cdx.json`
   - `COMPLIANCE_CHECKLIST.md`
   - `LICENSE`
   - `NOTICE.txt`
   - Dependency manifests
   - CI workflow evidence (`ci.yml`)

4. `04_Architecture_OnePager/`
   - `ARCHITECTURE_ONE_PAGER.md` (updated architecture summary)

## Verification Snapshot

- Test run status: `216 passed, 0 failed`
- SBOM generated: `CycloneDX JSON`
- Compliance checklist included: `Yes`
- Architecture one-pager included: `Yes`
