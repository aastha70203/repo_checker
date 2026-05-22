"""
test_cloner.py — Full edge-case test suite for RepoCloner.
Run with: python test_cloner.py
"""

import sys
from agent.cloner import RepoCloner

PASS = "✅ PASS"
FAIL = "❌ FAIL"

def run_test(label, url, expect_success, expect_warning_contains=None, expect_error_contains=None):
    print(f"\n{'─'*60}")
    print(f"TEST : {label}")
    print(f"URL  : {url}")
    print(f"{'─'*60}")

    cloner = RepoCloner()
    result = cloner.clone(url, progress_callback=lambda m: print(f"  → {m}"))
    print(result.summary())

    passed = True

    # Check success/failure
    if result.success != expect_success:
        print(f"  Expected success={expect_success}, got {result.success}")
        passed = False

    # Check warning message contains expected substring
    if expect_warning_contains:
        matched = any(expect_warning_contains.lower() in w.lower() for w in result.warnings)
        if not matched:
            print(f"  Expected warning containing '{expect_warning_contains}', got: {result.warnings}")
            passed = False
        else:
            print(f"  ✓ Warning correctly contains '{expect_warning_contains}'")

    # Check error message contains expected substring
    if expect_error_contains:
        if not result.error or expect_error_contains.lower() not in result.error.lower():
            print(f"  Expected error containing '{expect_error_contains}', got: {result.error}")
            passed = False
        else:
            print(f"  ✓ Error correctly contains '{expect_error_contains}'")

    if result.success:
        print(f"  📁 File tree (first 8):")
        for f in result.file_tree[:8]:
            print(f"     {f}")

    print(f"\n  {PASS if passed else FAIL}")
    cloner.cleanup()
    return passed


def run_cache_test():
    """Clone the same URL twice — second should be instant from cache."""
    print(f"\n{'─'*60}")
    print(f"TEST : Session cache — same URL cloned twice")
    print(f"{'─'*60}")

    cloner = RepoCloner()
    url = "https://github.com/karpathy/micrograd"

    import time
    t0 = time.time()
    r1 = cloner.clone(url, progress_callback=lambda m: print(f"  [1] → {m}"))
    t1 = time.time() - t0

    t0 = time.time()
    r2 = cloner.clone(url, progress_callback=lambda m: print(f"  [2] → {m}"))
    t2 = time.time() - t0

    print(f"  First  clone : {t1:.2f}s  (from_cache={r1.from_cache})")
    print(f"  Second clone : {t2:.2f}s  (from_cache={r2.from_cache})")

    passed = r1.success and r2.success and r2.from_cache and t2 < 0.5
    print(f"  ✓ Second clone was instant from cache" if passed else "  Second clone was NOT served from cache")
    print(f"\n  {PASS if passed else FAIL}")
    cloner.cleanup()
    return passed


if __name__ == "__main__":
    results = []

    # ── Valid inputs ─────────────────────────────────────────────────────────
    results.append(run_test(
        "Valid repo — full HTTPS URL",
        "https://github.com/karpathy/micrograd",
        expect_success=True,
    ))

    results.append(run_test(
        "Valid repo — trailing slash",
        "https://github.com/karpathy/micrograd/",
        expect_success=True,
    ))

    results.append(run_test(
        "Valid repo — .git suffix",
        "https://github.com/karpathy/micrograd.git",
        expect_success=True,
    ))

    results.append(run_test(
        "Valid repo — short owner/repo format",
        "karpathy/micrograd",
        expect_success=True,
    ))

    # ── Subdirectory / sub-page URLs ─────────────────────────────────────────
    results.append(run_test(
        "Subdirectory URL — /tree/main/... (should strip to root)",
        "https://github.com/karpathy/micrograd/tree/main/micrograd",
        expect_success=True,
        expect_warning_contains="sub-page",
    ))

    results.append(run_test(
        "Blob URL — /blob/main/file.py (should strip to root)",
        "https://github.com/karpathy/micrograd/blob/main/micrograd/engine.py",
        expect_success=True,
        expect_warning_contains="sub-page",
    ))

    # ── SSH URLs ─────────────────────────────────────────────────────────────
    results.append(run_test(
        "SSH URL — git@github.com:... (expect helpful redirect)",
        "git@github.com:karpathy/micrograd.git",
        expect_success=False,
        expect_error_contains="https://github.com/karpathy/micrograd",
    ))

    # ── Non-GitHub hosts ─────────────────────────────────────────────────────
    results.append(run_test(
        "GitLab URL (expect rejection)",
        "https://gitlab.com/gitlab-org/gitlab",
        expect_success=False,
        expect_error_contains="Only GitHub",
    ))

    # ── Non-existent repo ────────────────────────────────────────────────────
    results.append(run_test(
        "Non-existent repo",
        "https://github.com/this-user-xyz-fake/repo-does-not-exist-abc",
        expect_success=False,
        expect_error_contains="not found",
    ))

    # ── Totally invalid input ────────────────────────────────────────────────
    results.append(run_test(
        "Completely invalid input",
        "not-a-url-at-all",
        expect_success=False,
        expect_error_contains="valid GitHub",
    ))

    results.append(run_test(
        "Empty string",
        "",
        expect_success=False,
        expect_error_contains="valid GitHub",
    ))

    # ── Session cache ────────────────────────────────────────────────────────
    results.append(run_cache_test())

    # ── Summary ──────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    print(f"\n{'═'*60}")
    print(f"RESULTS: {passed}/{total} tests passed")
    print('═'*60)
    sys.exit(0 if passed == total else 1)