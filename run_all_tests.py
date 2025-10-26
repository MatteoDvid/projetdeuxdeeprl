"""
Master test script - Runs ALL tests sequentially
Final verification before training
"""

import subprocess
import sys

tests = [
    ("Setup Verification", "test_setup.py"),
    ("Import Tests", "test_imports.py"),
    ("Training Loop", "test_train.py"),
    ("Checkpoint System", "test_checkpoint.py"),
    ("Evaluation System", "test_evaluate.py"),
]

print("=" * 70)
print("RUNNING ALL TESTS - FINAL VERIFICATION")
print("=" * 70)
print()

all_passed = True
results = []

for test_name, test_file in tests:
    print(f"\n{'=' * 70}")
    print(f"TEST: {test_name}")
    print(f"File: {test_file}")
    print("=" * 70)

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode == 0:
            print(f"[PASS] {test_name}: PASSED")
            results.append((test_name, "PASS", None))
        else:
            print(f"[FAIL] {test_name}: FAILED")
            print(f"Error output:\n{result.stderr}")
            results.append((test_name, "FAIL", result.stderr))
            all_passed = False

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {test_name}: TIMEOUT")
        results.append((test_name, "TIMEOUT", None))
        all_passed = False
    except Exception as e:
        print(f"[ERROR] {test_name}: ERROR - {e}")
        results.append((test_name, "ERROR", str(e)))
        all_passed = False

# Summary
print("\n\n" + "=" * 70)
print("FINAL TEST SUMMARY")
print("=" * 70)

for test_name, status, error in results:
    symbol = "[OK]" if status == "PASS" else "[X]"
    print(f"{symbol} {test_name:.<50} {status}")

print("=" * 70)

if all_passed:
    print("\n>>> ALL TESTS PASSED <<<")
    print("\nThe project is FULLY FUNCTIONAL and ready for training!")
    print("\nTo start training:")
    print("  python train.py")
    sys.exit(0)
else:
    print("\n>>> SOME TESTS FAILED <<<")
    print("\nPlease review the errors above before training.")
    sys.exit(1)
