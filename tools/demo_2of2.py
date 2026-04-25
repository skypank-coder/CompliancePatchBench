"""
DEMONSTRATION: 2/2 Violations Fixed with Task1b
Shows multi-step success with connected violations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.patch_env import CompliancePatchEnv
from environment.tasks.task1b_connected import get_task

print("=" * 70)
print("DEMONSTRATION: 2/2 Violations Fixed (Multi-Step Success)")
print("=" * 70)
print()

task = get_task()
env = CompliancePatchEnv()

# Reset
obs = env.reset('task1b', task['codebase'], task['ground_truth'])
print(f"1. RESET")
print(f"   Violations: {obs['violations_total']}")
print(f"   Task: {task['description']}")
print()

# Read file
obs, r, done, info = env.step({'action_type': 'read_file', 'path': 'auth.py'})
print(f"2. READ FILE")
print(f"   File: auth.py")
print()

# Fix VIOLATION 1: GDPR-ART5-1A (line 19 - logging email)
# Original: logger.info(f"Login attempt for {email}")
# Fix: Remove email from log
patch1 = '    logger.info("Login attempt received")'

obs, r, done, info = env.step({
    'action_type': 'write_patch',
    'file': 'auth.py',
    'line_start': 19,
    'line_end': 19,
    'new_code': patch1
})
print(f"3. APPLY PATCH 1 (GDPR-ART5-1A)")
print(f"   Result: {obs['action_result']}")
print()

# Fix VIOLATION 2: GDPR-ART5-1C (line 24 - exposing password_hash)
# Original: return jsonify({'user': {'id': user.id, 'email': user.email, 'password_hash': user.password_hash}})
# Fix: Remove password_hash
patch2 = "        return jsonify({'user': {'id': user.id, 'email': user.email}})"

obs, r, done, info = env.step({
    'action_type': 'write_patch',
    'file': 'auth.py',
    'line_start': 24,
    'line_end': 24,
    'new_code': patch2
})
print(f"4. APPLY PATCH 2 (GDPR-ART5-1C)")
print(f"   Result: {obs['action_result']}")
print()

# Run CI
obs, r, done, info = env.step({'action_type': 'run_ci'})
print(f"5. RUN CI")
print(f"   {obs['action_result']}")
print(f"   Reward: {r:+.4f}")
print(f"   Fixed: {obs['violations_fixed']}/{obs['violations_total']}")
print(f"   Semantic validations passed: {obs['semantic_validations_passed']}")
print(f"   Semantic validations failed: {obs['semantic_validations_failed']}")
print()

# Finalize (if not already done)
if not done:
    obs, r, done, info = env.step({'action_type': 'finalize_patch'})
    print(f"6. FINALIZE")
    print(f"   Final score: {info['final_score']:.4f}")
    print(f"   Violations fixed: {info['critique']['violations_fixed']}/{info['critique']['violations_total']}")
    print()
else:
    # Episode ended automatically
    print(f"6. AUTO-FINALIZE (all violations fixed)")
    print(f"   Final score: {obs['cumulative_reward']:.4f}")
    print(f"   Violations fixed: {obs['violations_fixed']}/{obs['violations_total']}")
    print()
    info = {'critique': {'violations_fixed': obs['violations_fixed'], 'violations_total': obs['violations_total']}, 'final_score': obs['cumulative_reward']}

print("=" * 70)
if info['critique']['violations_fixed'] == 2:
    print("SUCCESS: 2/2 VIOLATIONS FIXED!")
    print("  - Multi-step success demonstrated")
    print("  - Connected violations handled")
    print(f"  - Final reward: {info['final_score']:.4f}")
elif info['critique']['violations_fixed'] > 0:
    print(f"PARTIAL: {info['critique']['violations_fixed']}/2 violations fixed")
    print("  - System works, patches need adjustment")
else:
    print("No violations fixed yet")
    print("  - Check patch alignment with execution tests")
print("=" * 70)
