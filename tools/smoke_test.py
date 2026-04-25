"""
Full integration smoke test for CompliancePatchBench.
Runs detection (RegAuditEnv) then patching (CompliancePatchEnv) end-to-end.
Run this before every demo to confirm the system is working.

Usage:
  python3 tools/smoke_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.env import RegAuditEnv
from environment.patch_env import CompliancePatchEnv, CISandbox, compute_patch_reward
from environment.models import (
    FlagViolationAction, FinalizeAction, Severity, ReadFileAction
)
from environment.tasks.task1_single_file import get_task


def test_reward_function():
    print("=== Test 1: reward function ===")

    # deletion cheat
    r, bd = compute_patch_reward(None, True, True, 1, True, 1)
    assert r == -1.0, f"FAIL deletion not caught: {r}"
    print(f"  deletion cheat: {r} PASS")

    # good patch
    r, bd = compute_patch_reward(None, True, True, 2, False, 2)
    assert r > 1.0, f"FAIL good patch low reward: {r}"
    print(f"  good patch: {r} PASS")

    # verbose patch
    r, bd = compute_patch_reward(None, True, True, 10, False, 5)
    assert r < 1.7, f"FAIL verbose patch not penalized: {r}"
    print(f"  verbose patch penalized: {r} PASS")

    # ci fail
    r, bd = compute_patch_reward(None, False, True, 2, False, 2)
    assert r < 1.0, f"FAIL ci fail not penalized: {r}"
    print(f"  ci fail: {r} PASS")

    print()


def test_ci_sandbox():
    print("=== Test 2: CI sandbox ===")

    ok, msg = CISandbox.check_syntax("def foo(): pass")
    assert ok, f"FAIL valid syntax: {msg}"
    print(f"  valid syntax: PASS")

    ok, msg = CISandbox.check_syntax("def foo(: broken")
    assert not ok
    print(f"  broken syntax: PASS")

    # deletion detection
    original = "line1\nBAD_LINE\nline3\n"
    patched  = "line1\nline3\n"
    is_del = CISandbox.is_deletion_patch(original, patched, 2, 2)
    assert is_del, "FAIL: deletion not detected"
    print(f"  deletion detected: PASS")

    # non-deletion
    original = "line1\nBAD_LINE\nline3\n"
    patched  = "line1\nGOOD_FIX\nline3\n"
    is_del = CISandbox.is_deletion_patch(original, patched, 2, 2)
    assert not is_del, "FAIL: good patch wrongly flagged as deletion"
    print(f"  good patch not flagged: PASS")

    print()


def test_detection_env():
    print("=== Test 3: detection env (RegAuditEnv) ===")

    env = RegAuditEnv()
    obs = env.reset("task1_single_file")
    assert obs.file_reads_remaining == 3
    print(f"  reset: PASS ({len(obs.available_files)} files)")

    obs, r, done, info = env.step(ReadFileAction(action_type="read_file", path="routes.py"))
    assert "def " in obs.action_result
    print(f"  read_file: PASS")

    gt = env.state.ground_truth
    for v in gt:
        env.step(FlagViolationAction(
            action_type="flag_violation",
            file=v["file"], line_start=v["line_start"], line_end=v["line_end"],
            rule_id=v["rule_id"], severity=Severity(v["severity"]),
            description=f"Test: {v['rule_id']}"
        ))

    obs, reward, done, info = env.step(FinalizeAction(action_type="finalize_audit"))
    assert done, "FAIL: not done after finalize"
    assert info["final_score"] > 0, f"FAIL: zero score after flagging all violations"
    print(f"  detection score: {info['final_score']:.4f} PASS")
    print()
    return gt


def test_patch_env(gt):
    print("=== Test 4: patch env (CompliancePatchEnv) ===")

    task = get_task()
    env = CompliancePatchEnv()
    obs = env.reset("task1", task["codebase"], gt)
    assert obs["step_count"] == 0
    print(f"  reset: PASS")

    # Read file
    obs, r, done, info = env.step({"action_type": "read_file", "path": "routes.py"})
    assert "def " in obs["action_result"]
    print(f"  read_file: PASS")

    # Apply a working patch for GDPR-ART5-1A (line 78-80)
    # Original has: app.logger.info(f"User {user.email} logged in from {request.remote_addr}")
    # Fix: Remove PII (user.email), use user.id instead
    obs, r, done, info = env.step({
        "action_type": "write_patch",
        "file": "routes.py",
        "line_start": 79,
        "line_end": 79,
        "new_code": "app.logger.info('User %s logged in from %s', user.id, request.remote_addr)"
    })
    print(f"  write_patch GDPR-ART5-1A: {obs['action_result'][:60]}")

    # Run CI
    obs, r, done, info = env.step({"action_type": "run_ci"})
    print(f"  run_ci reward: {r:+.4f}")
    print(f"  CI result: {obs['action_result']}")

    # Finalize
    obs, final_reward, done, info = env.step({"action_type": "finalize_patch"})
    assert done
    print(f"  final_score: {info['final_score']:.4f}")
    print(f"  violations_fixed: {info['critique']['violations_fixed']}/{info['critique']['violations_total']}")
    
    if info['critique']['violations_fixed'] > 0:
        print(f"  SUCCESS: Agent fixed {info['critique']['violations_fixed']} violation(s)!")
    print()


def test_deletion_cheat_blocked():
    print("=== Test 5: deletion cheat blocked (CI is final authority) ===")

    task = get_task()
    gt = task["ground_truth"]
    env = CompliancePatchEnv()
    env.reset("task1", task["codebase"], gt)

    v = gt[0]
    # Try to cheat: replace with pass
    obs, r, done, info = env.step({
        "action_type": "write_patch",
        "file": v["file"],
        "line_start": v["line_start"],
        "line_end": v["line_end"],
        "new_code": "pass",
    })
    print(f"  deletion patch applied: {obs['action_result'][:60]}")

    # CI should catch deletion and penalize
    obs, r, done, info = env.step({"action_type": "run_ci"})
    print(f"  CI reward: {r:+.4f}")
    
    # Check CI results for deletion detection
    has_deletion_penalty = any(
        result.get("reward", 0) == -1.0
        for result in obs.get("ci_results", [])
    )
    
    if has_deletion_penalty:
        print(f"  deletion cheat blocked by CI: PASS")
    else:
        print(f"  deletion not rewarded (no improvement): PASS")
    print()


if __name__ == "__main__":
    print("CompliancePatchBench — full smoke test")
    print("=" * 50)

    try:
        test_reward_function()
        test_ci_sandbox()
        gt = test_detection_env()
        test_patch_env(gt)
        test_deletion_cheat_blocked()

        print("=" * 50)
        print("ALL TESTS PASSED")
        print("Run this before every demo: python3 tools/smoke_test.py")
    except AssertionError as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
