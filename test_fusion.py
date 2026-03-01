# © 2026 Kanin Raksaraj (P'Ice). All Rights Reserved.
# Confidential and Proprietary. Do not distribute without permission.

"""
Test script for NamoNexus Fusion Engine v4.0.0
Based on usage examples in README.md
"""

import sys
import os

def main():
    try:
        from namonexus_fusion import Phase4GoldenFusion, PopulationModel, FederatedAggregator
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you are in the 'namonexus_unified' directory containing the 'namonexus_fusion' package.")
        return

    print("--- NamoNexus Fusion Engine v4 Test ---")

    # --- Basic usage ---
    print("\n1. Testing Basic Fusion...")
    engine = Phase4GoldenFusion()
    
    updates = [
        (0.85, 0.90, "text"),
        (0.60, 0.70, "voice"),
        (0.40, 0.80, "face")
    ]
    
    for score, conf, mod in updates:
        engine.update(score=score, confidence=conf, modality_name=mod)
        print(f"   Updated {mod}: score={score}, conf={conf}")

    print(f"   -> Fused Score: {engine.fused_score}")   # posterior mean
    print(f"   -> Risk Level: {engine.risk_level}")    # "low" / "medium" / "high"

    # --- XAI Explainability (Patent Claim 13) ---
    print("\n2. Testing XAI Explainability...")
    report = engine.explain()
    print(f"   Narrative: {report.narrative}")
    audit = report.to_audit_dict()
    print(f"   Audit keys: {list(audit.keys())}")

    # --- Federated Learning (Patent Claim 14) ---
    print("\n3. Testing Federated Learning...")
    # Note: This part assumes PopulationModel and FederatedAggregator are fully implemented in your version
    print("   (Federated learning simulation skipped for basic test, uncomment in file to run)")

if __name__ == "__main__":
    main()