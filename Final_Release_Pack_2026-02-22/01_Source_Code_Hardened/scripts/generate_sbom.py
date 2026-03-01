#!/usr/bin/env python
"""
Generate a lightweight CycloneDX-style SBOM from requirements.txt.

Usage:
    python scripts/generate_sbom.py --requirements requirements.txt --output sbom.cdx.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional


SPDX_HINTS = {
    "mit license": "MIT",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd 2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "bsd 3-clause": "BSD-3-Clause",
    "bsd license": "BSD-3-Clause",
    "apache license, version 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "mpl-2.0": "MPL-2.0",
    "lgpl": "LGPL-3.0-or-later",
    "gpl": "GPL-3.0-or-later",
    "isc license": "ISC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SBOM from pip requirements")
    parser.add_argument("--requirements", default="requirements.txt", help="Path to requirements file")
    parser.add_argument("--output", default="sbom.cdx.json", help="Output SBOM path")
    return parser.parse_args()


def normalize_name(req_line: str) -> str:
    base = req_line.split(";", 1)[0].strip()
    base = re.split(r"[<>=!~]", base, maxsplit=1)[0].strip()
    return base.split("[", 1)[0].strip()


def parse_requirements(path: Path) -> List[str]:
    packages: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r") or line.startswith("--"):
            continue
        name = normalize_name(line)
        if name:
            packages.append(name)
    return packages


def _classifiers(meta: metadata.PackageMetadata) -> List[str]:
    return [value for key, value in meta.items() if key == "Classifier"]


def _spdx_from_text(text: str) -> Optional[str]:
    lowered = text.lower()
    for hint, spdx in SPDX_HINTS.items():
        if hint in lowered:
            return spdx
    return None


def detect_licenses(meta: metadata.PackageMetadata) -> List[Dict[str, Dict[str, str]]]:
    # Prefer Trove classifiers over the free-form "License" field.
    for classifier in _classifiers(meta):
        if not classifier.startswith("License ::"):
            continue
        spdx = _spdx_from_text(classifier)
        if spdx:
            return [{"license": {"id": spdx}}]
        return [{"license": {"name": classifier}}]

    license_text = (meta.get("License") or "").strip()
    if license_text and license_text.lower() not in {"unknown", "n/a"}:
        tokens = re.split(r"\s+or\s+|\s+\|\s+|/", license_text, flags=re.IGNORECASE)
        parsed: List[Dict[str, Dict[str, str]]] = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            spdx = _spdx_from_text(token)
            if spdx:
                parsed.append({"license": {"id": spdx}})

        if parsed:
            # Keep unique ids only.
            unique = []
            seen = set()
            for item in parsed:
                lid = item["license"].get("id")
                if lid and lid not in seen:
                    unique.append(item)
                    seen.add(lid)
            return unique

        shortened = license_text if len(license_text) <= 160 else f"{license_text[:157]}..."
        return [{"license": {"name": shortened}}]

    return [{"license": {"name": "UNKNOWN"}}]


def build_component(package_name: str) -> Dict[str, object]:
    try:
        dist = metadata.distribution(package_name)
        version = dist.version
        meta = dist.metadata
        license_objs = detect_licenses(meta)
    except metadata.PackageNotFoundError:
        version = "UNRESOLVED"
        license_objs = [{"license": {"name": "UNRESOLVED"}}]
        meta = {}  # type: ignore[assignment]

    canonical = package_name.replace("_", "-").lower()
    return {
        "type": "library",
        "name": package_name,
        "version": version,
        "licenses": license_objs,
        "purl": f"pkg:pypi/{canonical}@{version}",
        "supplier": {"name": meta.get("Author", "UNKNOWN")} if hasattr(meta, "get") else {"name": "UNKNOWN"},
    }


def generate_sbom(requirements: Path, output: Path) -> None:
    packages = parse_requirements(requirements)
    components = [build_component(name) for name in packages]

    doc = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "tools": [{"vendor": "NamoNexus", "name": "generate_sbom.py"}],
            "component": {
                "type": "application",
                "name": "namonexus-fusion",
                "version": os.getenv("NAMONEXUS_VERSION", "4.0.0"),
            },
        },
        "components": components,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, indent=2), encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    generate_sbom(Path(args.requirements), Path(args.output))
