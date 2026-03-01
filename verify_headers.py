# © 2026 Kanin Raksaraj (P'Ice). All Rights Reserved.
# Confidential and Proprietary. Do not distribute without permission.

import os
import ast

REQUIRED_HEADER = "# © 2026 Kanin Raksaraj (P'Ice). All Rights Reserved."

def verify_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Check for Header
    if REQUIRED_HEADER not in content:
        print(f"❌ MISSING HEADER: {filepath}")
        return False
    
    # 2. Check Syntax (ensure header didn't break python)
    try:
        ast.parse(content)
    except SyntaxError as e:
        print(f"❌ SYNTAX ERROR: {filepath} - {e}")
        return False
        
    print(f"✅ VERIFIED: {filepath}")
    return True

def main():
    print("🔍 Starting Copyright Header Verification...")
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    all_passed = True
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py") and file != "verify_headers.py":
                if not verify_file(os.path.join(root, file)):
                    all_passed = False
                    
    if all_passed:
        print("\n✨ ALL FILES PASSED VERIFICATION. READY FOR DEPLOYMENT.")
    else:
        print("\n⚠️ SOME FILES FAILED VERIFICATION.")

if __name__ == "__main__":
    main()