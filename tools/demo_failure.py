"""
Failure Demonstration: Anti-Cheat in Action
Shows what happens when agent attempts to cheat via deletion.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.patch_env import CompliancePatchEnv
from environment.tasks.task1_single_file import get_task

print("=" * 70)
print("FAILURE DEMONSTRATION: Anti-Cheat System")
print("=" * 70)
print()
print("This demo shows what happens when an agent attempts to cheat")
print("by deleting violations instead of fixing them properly.")
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

# Attempt 1: Complete deletion (empty string)
print(f"3. ATTEMPT 1: Complete Deletion")
print(f"   Strategy: Replace violation with empty string")
print()

obs, r, done, info = env.step({
    'action_type': 'write_patch',
    'file': 'routes.py',
    'line_start': 74,
    'line_end': 74,
    'new_code': ''
})

print(f"   Result: {obs['action_result']}")
print(f"   Reward: {r:+.4f}")
print()

# Attempt 2: Trivial replacement (pass statement)
print(f"4. ATTEMPT 2: Trivial Replacement")
print(f"   Strategy: Replace with 'pass' statement")
print()

obs, r, done, info = env.step({
    'action_type': 'write_patch',
    'file': 'routes.py',
    'line_start': 74,
    'line_end': 74,
    'new_code': '    pass'
})

print(f"   Result: {obs['action_result']}")
print(f"   Reward: {r:+.4f}")
print()

# Attempt 3: Comment-only replacement
print(f"5. ATTEMPT 3: Comment-Only Replacement")
print(f"   Strategy: Replace with comment")
print()

obs, r, done, info = env.step({
    'action_type': 'write_patch',
    'file': 'routes.py',
    'line_start': 74,
    'line_end': 74,
    'new_code': '    # TODO: Fix this violation'
})

print(f"   Result: {obs['action_result']}")
print(f"   Reward: {r:+.4f}")
print()

# Run CI to see deletion detection
print(f"6. RUN CI (Deletion Detection)")
print()

obs, r, done, info = env.step({'action_type': 'run_ci'})

print(f"   CI Results:")
print(f"   {obs['action_result']}")
print(f"   Reward: {r:+.4f}")
print(f"   Violations fixed: {obs['violations_fixed']}/{obs['violations_total']}")
print()

# Show CI details
if obs['ci_results']:
    print(f"   Detailed CI Output:")
    for result in obs['ci_results']:
        if result.get('deletion_blocked'):
            print(f"   - {result['rule_id']}: {result['reason']}")
            print(f"     Deletion blocked: YES")
            print(f"     Reward: {result['reward']:+.4f}")
print()

# Now show proper fix
print(f"7. PROPER FIX (Recovery)")
print(f"   Strategy: Minimal semantic fix")
print()

obs, r, done, info = env.step({
    'action_type': 'write_patch',
    'file': 'routes.py',
    'line_start': 74,
    'line_end': 74,
    'new_code': '    app.logger.info("User %s logged in from %s", user.id, request.remote_addr)'
})

print(f"   Result: {obs['action_result']}")
print(f"   Reward: {r:+.4f}")
print()

# Run CI again
obs, r, done, info = env.step({'action_type': 'run_ci'})

print(f"8. RUN CI (After Proper Fix)")
print(f"   {obs['action_result']}")
print(f"   Reward: {r:+.4f}")
print(f"   Violations fixed: {obs['violations_fixed']}/{obs['violations_total']}")
print()

print("=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print()
print("1. DELETION ATTEMPTS BLOCKED:")
print("   - Empty string: Rejected or penalized")
print("   - Pass statement: Detected as trivial")
print("   - Comment-only: Detected as non-semantic")
print()
print("2. MULTI-LAYER DETECTION:")
print("   - Layer 1: Line count reduction")
print("   - Layer 2: Empty replacement")
print("   - Layer 3: Comment-only code")
print("   - Layer 4: Trivial statements")
print("   - Layer 5: AST semantic preservation")
print("   - Final: CI double-check")
print()
print("3. EXPLICIT MESSAGING:")
print("   - CI clearly states 'DELETION DETECTED'")
print("   - Agents receive clear feedback")
print("   - No ambiguity about what went wrong")
print()
print("4. RECOVERY POSSIBLE:")
print("   - Agent can try again with proper fix")
print("   - Proper fixes are rewarded positively")
print("   - System encourages learning, not just blocking")
print()
print("=" * 70)
print("ANTI-CHEAT SYSTEM VALIDATED")
print("=" * 70)
