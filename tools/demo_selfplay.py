"""
Self-Play Demonstration: Patcher vs Adversary
Shows the adversarial loop where adversary generates violations
and patcher learns to fix them.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.adversary import AdversaryAgent
from environment.patch_env import CompliancePatchEnv
from environment.tasks.task1_single_file import get_task

print("=" * 70)
print("SELF-PLAY DEMONSTRATION: Patcher vs Adversary")
print("=" * 70)
print()

# Initialize agents
adversary = AdversaryAgent(seed=42)
env = CompliancePatchEnv()

# Track performance across rounds
performance_history = []

# Run 3 rounds of self-play
for round_num in range(1, 4):
    print(f"\n{'='*70}")
    print(f"ROUND {round_num}")
    print(f"{'='*70}\n")
    
    # Adversary evaluates patcher performance and adjusts difficulty
    if round_num == 1:
        difficulty = "easy"
        print(f"Adversary: Starting with EASY violations")
    else:
        prev_perf = performance_history[-1]
        difficulty = adversary.evaluate_patcher_performance(
            prev_perf["violations_fixed"],
            prev_perf["violations_total"],
            prev_perf["reward"]
        )
        print(f"Adversary: Patcher performance = {prev_perf['violations_fixed']}/{prev_perf['violations_total']}")
        print(f"Adversary: Escalating to {difficulty.upper()} violations")
    
    print()
    
    # Adversary generates new violations
    print(f"Adversary: Generating {difficulty} violations...")
    violations = []
    for i, rule_id in enumerate(["GDPR-ART5-1A", "GDPR-ART5-1C"], 1):
        code, metadata = adversary.generate_violation(rule_id, difficulty)
        violations.append({
            "file": "routes.py",
            "rule_id": rule_id,
            "severity": "high",
            "line_start": 70 + i * 5,
            "line_end": 70 + i * 5,
        })
        print(f"  {i}. {rule_id}: {metadata['description']}")
        print(f"     Code: {code}")
    
    print()
    
    # Patcher attempts to fix violations
    print(f"Patcher: Analyzing violations...")
    task = get_task()
    
    # Use adversary-generated violations
    obs = env.reset('task1_selfplay', task['codebase'], violations)
    
    print(f"Patcher: Found {obs['violations_total']} violations")
    print(f"Patcher: Attempting fixes...")
    print()
    
    # Simulate patcher actions (in real training, this would be the trained model)
    # For demo, we'll use heuristic fixes
    
    # Read file
    obs, r, done, info = env.step({'action_type': 'read_file', 'path': 'routes.py'})
    
    # Apply patches (simplified for demo)
    patches_applied = 0
    
    # Fix GDPR-ART5-1A (if present)
    if any(v['rule_id'] == 'GDPR-ART5-1A' for v in violations):
        obs, r, done, info = env.step({
            'action_type': 'write_patch',
            'file': 'routes.py',
            'line_start': 74,
            'line_end': 74,
            'new_code': '    app.logger.info("User %s logged in from %s", user.id, request.remote_addr)'
        })
        if "applied" in obs['action_result'].lower():
            patches_applied += 1
            print(f"  Patch 1: Applied (GDPR-ART5-1A)")
        else:
            print(f"  Patch 1: Failed - {obs['action_result']}")
    
    # Run CI
    obs, r, done, info = env.step({'action_type': 'run_ci'})
    
    print()
    print(f"CI Results:")
    print(f"  Violations fixed: {obs['violations_fixed']}/{obs['violations_total']}")
    print(f"  Reward: {r:+.4f}")
    print(f"  Semantic validations passed: {obs['semantic_validations_passed']}")
    
    # Record performance
    performance_history.append({
        "round": round_num,
        "difficulty": difficulty,
        "violations_total": obs['violations_total'],
        "violations_fixed": obs['violations_fixed'],
        "reward": r,
        "semantic_validations_passed": obs['semantic_validations_passed'],
    })
    
    print()
    
    # Adversary responds
    if obs['violations_fixed'] == obs['violations_total']:
        print(f"Adversary: All violations fixed! Escalating difficulty...")
    elif obs['violations_fixed'] > 0:
        print(f"Adversary: Partial success. Adapting strategy...")
    else:
        print(f"Adversary: No fixes. Maintaining difficulty...")

print()
print("=" * 70)
print("SELF-PLAY SUMMARY")
print("=" * 70)
print()

# Summary table
print(f"{'Round':<8} {'Difficulty':<12} {'Fixed':<10} {'Reward':<10} {'Validations':<12}")
print("-" * 70)
for perf in performance_history:
    print(f"{perf['round']:<8} {perf['difficulty']:<12} {perf['violations_fixed']}/{perf['violations_total']:<8} {perf['reward']:+.4f}    {perf['semantic_validations_passed']:<12}")

print()
print("=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print()
print("1. ADAPTIVE CURRICULUM:")
print("   Adversary adjusts difficulty based on patcher performance")
print()
print("2. SELF-IMPROVEMENT LOOP:")
print("   Patcher -> Fix -> Adversary -> Harder Violations -> Patcher")
print()
print("3. CONTINUOUS CHALLENGE:")
print("   As patcher improves, adversary escalates difficulty")
print()
print("4. MEASURABLE PROGRESS:")
print("   Reward and fix rate track patcher capability over rounds")
print()
print("=" * 70)
print("SELF-PLAY DEMONSTRATION COMPLETE")
print("=" * 70)
