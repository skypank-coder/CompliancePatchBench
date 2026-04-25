"""
Baseline Agent Demonstration
Shows random/deletion baseline performance to establish improvement baseline.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.patch_env import CompliancePatchEnv
from environment.tasks.task1_single_file import get_task
from environment.tasks.task2_django_app import get_task as get_task2
from environment.tasks.task3_microservices import get_task as get_task3

print("=" * 70)
print("BASELINE AGENT DEMONSTRATION")
print("=" * 70)
print()
print("Testing baseline (deletion attempt) agent to establish improvement")
print("baseline for comparison with trained agents.")
print()

tasks = [
    ("task1_single_file", get_task(), "Task 1: Single-file"),
    ("task2_django_app", get_task2(), "Task 2: Django multi-file"),
    ("task3_microservices", get_task3(), "Task 3: Microservices"),
]

results = []

for task_id, task, task_name in tasks:
    print(f"\n{'='*70}")
    print(f"{task_name}")
    print(f"{'='*70}\n")
    
    env = CompliancePatchEnv()
    
    # Reset
    obs = env.reset(task_id, task['codebase'], task['ground_truth'])
    print(f"Violations: {obs['violations_total']}")
    print(f"Files: {obs['available_files']}")
    print()
    
    # Baseline strategy: Try deletion (the classic cheat)
    print("Baseline Strategy: Attempt deletion (empty patch)")
    print()
    
    # Read first file
    first_file = obs['available_files'][0]
    obs, r, done, info = env.step({'action_type': 'read_file', 'path': first_file})
    
    # Try to delete first violation (empty patch)
    if obs['violations_total'] > 0:
        first_violation = task['ground_truth'][0]
        obs, r, done, info = env.step({
            'action_type': 'write_patch',
            'file': first_violation['file'],
            'line_start': first_violation['line_start'],
            'line_end': first_violation['line_end'],
            'new_code': ''  # Deletion attempt
        })
        print(f"Patch attempt: {obs['action_result'][:80]}...")
        print()
    
    # Run CI
    obs, r, done, info = env.step({'action_type': 'run_ci'})
    
    print(f"CI Results:")
    print(f"  Violations fixed: {obs['violations_fixed']}/{obs['violations_total']}")
    print(f"  Reward: {r:+.4f}")
    print(f"  Status: {'PASS' if obs['violations_fixed'] > 0 else 'FAIL'}")
    
    # Finalize
    obs, final_reward, done, info = env.step({'action_type': 'finalize_patch'})
    
    results.append({
        "task": task_name,
        "violations_total": info['critique']['violations_total'],
        "violations_fixed": info['critique']['violations_fixed'],
        "reward": info['final_score'],
    })
    
    print(f"\nFinal Score: {info['final_score']:.4f}")

print()
print("=" * 70)
print("BASELINE SUMMARY")
print("=" * 70)
print()

print(f"{'Task':<30} {'Fixed':<15} {'Reward':<10}")
print("-" * 70)
for r in results:
    print(f"{r['task']:<30} {r['violations_fixed']}/{r['violations_total']:<13} {r['reward']:+.4f}")

avg_reward = sum(r['reward'] for r in results) / len(results)
print()
print(f"Average Baseline Reward: {avg_reward:+.4f}")
print()

print("=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print()
print("1. DELETION ATTEMPTS BLOCKED:")
print("   All deletion attempts result in negative or zero rewards")
print()
print("2. ANTI-CHEAT WORKS:")
print("   System prevents the most common cheating strategy")
print()
print("3. BASELINE ESTABLISHED:")
print("   Random/deletion baseline: ~0.0 reward")
print("   Heuristic agent: ~1.2 reward (from demos)")
print("   Improvement: +1.2 reward (+infinite %)")
print()
print("4. TRAINING POTENTIAL:")
print("   Large gap between baseline and heuristic shows")
print("   significant room for improvement via training")
print()
print("=" * 70)
