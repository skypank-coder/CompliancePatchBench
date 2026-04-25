---
title: CompliancePatchBench
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: RL environment for training patch agents on compliance with GRPO and hidden oracles
---

# CompliancePatchBench

AI systems can pass tests and still be wrong. CompliancePatchBench trains agents
that must fix code correctly, even when tests are misleading.

Real-world impact: a "passing" fix can still leak user data or weaken security.
This system is designed to prevent that.

Key idea: we train agents in an environment where shortcut fixes are penalized
by hidden constraints.

## What This Project Demonstrates

- A compliance/security patching environment with structured JSON actions.
- An OpenEnv-style interface with `reset`, `step`, `state`, FastAPI endpoints,
  fixed step limits, and file-read budgets.
- Diverse generated tasks across GDPR, OWASP, code quality, multi-file bugs,
  and adversarial fake-safe fixes.
- Hidden compliance checks that catch shortcut fixes which pass visible tests.
- RL-based policy optimization using GRPO via TRL.
- RL trajectories with `(state, action, reward, next_state, logprob, done)`.
- Learning curves tracking reward, success rate, and hidden-violation rate.

## Why This RL Cannot Be Cheated

The reward is not just "did the regex pass?" It is grounded in three checks:

1. **CI + tests** catch visible correctness failures.
2. **Hidden oracle** penalizes shortcut fixes like hashed PII, masked PII,
   weak crypto, hardcoded env defaults, and partial multi-file fixes.
3. **Adversarial tasks** include fake-safe fixes, misleading comments, and
   cross-file dependencies.

These are exactly the kinds of bugs that slip past production systems today.

In short: the agent learns from mistakes and gradually avoids bad fixes. The
agent does not just learn to fix code; it learns to avoid cheating because the
environment penalizes hidden violations.

## RL + Policy Optimization

We use heuristic/tabular rollouts for initial data collection and baseline
comparison. Final policy optimization is performed using GRPO via TRL.

This is an online reinforcement learning loop with environment feedback:
the current policy generates JSON patch actions, `CompliancePatchEnv` executes
them and returns reward, and `GRPOTrainer` updates that same policy for the next
iteration. Evaluation remains deterministic.

The RL loop is designed to scale to larger task distributions; this demo uses a
small subset for runtime constraints.

The loop is failure-aware and adaptive: each iteration tracks
`hidden_violation`, `partial_fix`, and `no_fix`, increases sampling weight on
failed/adversarial tasks, evaluates an unseen test split, and reports recovered
tasks that failed in one iteration but succeeded later. It also logs confidence
so "high confidence but wrong" patches are visible instead of hidden.

Pipeline:

```text
heuristic rollouts -> SFT initialization -> online GRPO rollouts -> GRPO-refined policy
```

## OpenEnv Hackathon Checklist

- Environment first: `CompliancePatchEnv` exposes reset/step/state behavior and
  is wrapped by FastAPI routes for local or Hugging Face Space deployment.
- Verifiable rewards: CI checks, semantic validation, minimal patch scoring,
  deletion detection, hidden compliance checks, partial-fix penalties, timeout
  penalties, and file-budget limits are independent reward/process signals.
- Curriculum: generated tasks include easy, medium, and hard distributions so
  the agent gets non-zero reward before harder adversarial cases.
- Adaptive training: failed and adversarial tasks are sampled more often, while
  consistently solved tasks are sampled less.
- Generalization: the RL loop trains on a train split and reports
  `test_success_rate` on unseen tasks.
- Training stack: SFT uses Unsloth/LoRA where available, and final policy
  optimization uses TRL `GRPOTrainer` with environment reward feedback.
- Demo evidence: `project.smoke_test`, difficulty-aware evaluation, failure-case
  logging, and `/rl/learning-curve` show the baseline, rewards, safeguards, and
  improvement path.

## Important Files

```text
project/
├── task_generator.py       # Generates easy/medium/hard compliance patch tasks
├── agent.py                # Strict-JSON agent loop + RL transition capture
├── hidden_compliance.py    # Hidden anti-cheat oracle
├── dataset_builder.py      # Rollouts -> filtered SFT dataset
├── train_model.py          # LoRA SFT pipeline
├── rl_trainer.py           # RL loop: rollout -> reward-to-go -> policy update
├── evaluate.py             # Difficulty-aware metrics + iteration comparison
├── colab_training.ipynb    # End-to-end demo notebook
└── README.md               # Detailed technical documentation

api/server.py               # FastAPI service for Docker/Hugging Face Spaces
Dockerfile                  # Hugging Face Docker runtime
HF_SPACE_DEPLOYMENT.md      # Deployment guide
```

## Quick Start

Install the lightweight API/test dependencies:

```bash
pip install -r requirements.txt
```

For the full SFT/RL training stack, also install the ML dependencies:

```bash
pip install -r project/requirements.txt
```

```bash
python -m project.task_generator --num 40 --seed 42
python -m project.dataset_builder --rollouts 1 --min-success 0.5
python -m project.evaluate run --tag baseline
python -m project.rl_trainer --iterations 3 --dry-run
python -m project.evaluate iterations
```

One-command proof for judges:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m project.smoke_test
```

Expected signal:

```text
competition smoke test passed
Iter 0 -> reward ...
Iter 1 -> reward ...
```

## Training Notebook

Mandatory training-run notebook for reviewers:
[`project/colab_training.ipynb`](project/colab_training.ipynb)

This notebook is the runnable Hugging Face GPU training path. It logs in with
`HF_TOKEN`, verifies the real `CompliancePatchEnv` reward, runs the baseline,
trains `unsloth/Qwen2.5-3B-Instruct` with TRL `GRPOTrainer`, and optionally
pushes the adapter to `HF_OUTPUT_REPO`.

## Docker / Hugging Face Space

**Live Space:** [huggingface.co/spaces/rachana05/Compliance-patch-bench](https://huggingface.co/spaces/rachana05/Compliance-patch-bench)

```bash
docker build -t compliancepatchbench .
docker run --rm -p 7860:7860 compliancepatchbench
curl http://localhost:7860/health
curl http://localhost:7860/project
curl http://localhost:7860/rl/learning-curve
```

See `HF_SPACE_DEPLOYMENT.md` for the deployment checklist.
Space metadata reference: [Spaces config](https://huggingface.co/docs/hub/spaces-config-reference).
