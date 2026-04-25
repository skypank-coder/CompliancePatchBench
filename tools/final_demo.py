"""
FINAL DEMONSTRATION: CompliancePatchBench with WORKING patches.
Shows that the system CAN succeed and produce positive rewards.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.patch_env import CompliancePatchEnv
from environment.tasks.task1_single_file import get_task

print("=" * 70)
print("FINAL DEMONSTRATION: Successful Patch Application")
print("=" * 70)
print()

task = get_task()
env = CompliancePatchEnv()

# Reset
obs = env.reset('task1', task['codebase'], task['ground_truth'])
print(f"1. RESET")
print(f"   Violations: {obs['violations_total']}")
print(f"   Files: {obs['available_files']}")
print()

# Read file
obs, r, done, info = env.step({'action_type': 'read_file', 'path': 'routes.py'})
print(f"2. READ FILE")
print(f"   File read successfully")
print()

# Apply WORKING patch for GDPR-ART5-1A
# Line 74: app.logger.info(f"User {user.email} logged in from {request.remote_addr}")
# Fix: Remove PII, use user.id with proper indentation
working_patch = '    app.logger.info("User %s logged in from %s", user.id, request.remote_addr)'

obs, r, done, info = env.step({
    'action_type': 'write_patch',
    'file': 'routes.py',
    'line_start': 74,
    'line_end': 74,
    'new_code': working_patch
})
print(f"3. APPLY PATCH")
print(f"   Result: {obs['action_result']}")
print(f"   Reward: {r:+.4f}")
print()

# Run CI
obs, r, done, info = env.step({'action_type': 'run_ci'})
print(f"4. RUN CI")
print(f"   {obs['action_result']}")
print(f"   Reward: {r:+.4f}")
print(f"   Fixed: {obs['violations_fixed']}/{obs['violations_total']}")
print()

# Finalize
obs, r, done, info = env.step({'action_type': 'finalize_patch'})
print(f"5. FINALIZE")
print(f"   Final score: {info['final_score']:.4f}")
print(f"   Violations fixed: {info['critique']['violations_fixed']}/{info['critique']['violations_total']}")
print()

print("=" * 70)
if info['final_score'] > 0:
    print(f"SUCCESS: System demonstrates PROGRESS!")
    print(f"  - Violations fixed: {info['critique']['violations_fixed']}")
    print(f"  - Final reward: {info['final_score']:.4f}")
    print(f"  - Agent CAN succeed and learn!")
else:
    print(f"No positive reward yet")
    print(f"  - Check patch syntax and logic")
print("=" * 70)
