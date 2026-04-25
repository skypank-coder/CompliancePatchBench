"""
Demonstration: Successful patch trajectories in CompliancePatchBench.
Shows that agents CAN succeed with proper patches.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.patch_env import CompliancePatchEnv
from environment.tasks.task1_single_file import get_task

def demonstrate_successful_patch():
    print("=" * 70)
    print("DEMONSTRATION: Successful Patch Trajectory")
    print("=" * 70)
    print()
    
    task = get_task()
    env = CompliancePatchEnv()
    
    # Reset
    obs = env.reset('task1', task['codebase'], task['ground_truth'])
    print(f"1. RESET")
    print(f"   Violations to fix: {obs['violations_total']}")
    print(f"   Files: {obs['available_files']}")
    print()
    
    # Read file
    obs, r, done, info = env.step({'action_type': 'read_file', 'path': 'routes.py'})
    print(f"2. READ FILE")
    print(f"   Lines read: {len(obs['action_result'].split(chr(10)))}")
    print()
    
    # Apply a WORKING patch for GDPR-ART5-1A (line 78-80)
    # Original: app.logger.info(f"User {user.email} logged in from {request.remote_addr}")
    # Fix: Remove user.email, use user.id instead
    working_patch = """app.logger.info("User %s logged in from %s", user.id, request.remote_addr)
return jsonify({'token': 'fake-jwt-token', 'user_id': user.id})"""
    
    obs, r, done, info = env.step({
        'action_type': 'write_patch',
        'file': 'routes.py',
        'line_start': 79,
        'line_end': 80,
        'new_code': working_patch
    })
    print(f"3. APPLY PATCH (GDPR-ART5-1A)")
    print(f"   Result: {obs['action_result'][:60]}...")
    print(f"   Reward: {r:+.4f}")
    print()
    
    # Run CI
    obs, r, done, info = env.step({'action_type': 'run_ci'})
    print(f"4. RUN CI")
    print(f"   {obs['action_result']}")
    print(f"   Reward: {r:+.4f}")
    print(f"   Violations fixed: {obs['violations_fixed']}/{obs['violations_total']}")
    print()
    
    # Finalize
    obs, r, done, info = env.step({'action_type': 'finalize_patch'})
    print(f"5. FINALIZE")
    print(f"   Final score: {info['final_score']:.4f}")
    print(f"   Done: {done}")
    print()
    
    print("=" * 70)
    if info['final_score'] > 0:
        print("SUCCESS: Agent demonstrated positive reward trajectory!")
    else:
        print("No reward yet - patch needs adjustment")
    print("=" * 70)
    print()
    
    return info['final_score']

if __name__ == "__main__":
    score = demonstrate_successful_patch()
    sys.exit(0 if score >= 0 else 1)
