---
title: CompliancePatchBench
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
pinned: true
license: apache-2.0
short_description: RL environment that trains LLMs to fix security violations without cheating
---

# CompliancePatchBench 🛡️

*Trains LLMs to fix GDPR/OWASP violations in Python code — with a reward function that cannot be gamed.*

**Theme:** World Modeling 3.1 + Self-Improvement 4 · **Model:** Qwen2.5-3B · **Method:** GRPO + Unsloth · **Status:** Trained ✅

🤗 [Live Demo on HF Space](https://huggingface.co/spaces/skypank-coder/CompliancePatchBench) · 📓 [Colab Training Notebook](https://github.com/skypank-coder/CompliancePatchBench/blob/main/project/colab_training.ipynb)

## The Problem

**Capability gap:** Most teams still chain static analysis with manual edits. A model can be “logically” close to a fix while being catastrophically wrong for security and compliance: the bug disappears from a grep, but the product is worse, or the leak is just moved.

When you ask a vanilla LLM to fix a GDPR violation it does one of three things: it `deletes the flagged line` (violation gone, app broken, reward -1.0), or it `hashes the PII before logging` (still a violation — a hidden oracle catches that), or it `writes TODO: fix this` (passes no checker). None pass a real compliance audit. Semgrep and Bandit can find violations, but they do not apply fixes. CompliancePatchBench trains agents that do both — find and fix — correctly.

**What the agent must learn instead:** read only what the budget allows, write a *minimal* code change, run the same kind of checks a CI job would run, and finish with a final score that reflects *hidden* compliance properties — not a single string match.

**Who pays attention:** anyone shipping Python under GDPR or OWASP-style controls, where “green CI” and “clean Semgrep” are necessary but not sufficient for an audit. The benchmark is built so that a policy cannot ride on a brittle exploit; it has to show up in multi-signal reward.

## How It Works

CompliancePatchBench is **OpenEnv-compliant**: `reset()` / `step()` / `state()` plus a FastAPI server. Episodes are episodes: each reset hands the agent a task bundle (code, violation metadata, read budget, step cap). Each step is validated JSON, executed in a sandbox, and scored.

**What the agent sees:** structured observations — which files exist, which rules fired, where the violations are, and how many `read_file` calls remain. It does *not* get a free pass to dump the full tree on hard tasks: the read budget is real.

**What the agent does:** tool-style actions only — `read_file | write_patch | run_ci | finalize_patch` — in JSON, so the policy cannot improvise a silent side channel.

**What the agent is rewarded for:** a composite of CI outcome, regression behavior, edit size, extra churn, and a separate deletion channel (details in the next section). The hidden oracle is what stops “CI-only” cheating.

**Task ladder (5 training tasks, fixed IDs):** single Flask file (easy) → multi-file Django (medium) → additional multi-file and REST paths (hard) → 4-microservice system with **cross-file** dependencies (hard). Cross-file violations require **multi-hop reasoning** across services; the read budget is what forces *strategic* prioritization instead of brute-force reads.

```
┌─────────────────────────────────────────────────────────┐
│                  CompliancePatchBench                    │
│                                                         │
│  Codebase ──► Patcher Agent ──► CI Checker ──► Reward   │
│                    ▲                              │      │
│                    │                              ▼      │
│             Adversary Agent ◄── harder violations        │
└─────────────────────────────────────────────────────────┘
```

A typical episode (same interface the policy sees in training and eval):

```text
# Agent receives at episode start:
# File: routes.py (180 lines, Flask app)  
# Violation: GDPR-ART5-1A at line 74 — severity: high
# Read budget: 3 files remaining

# Step 1 — agent reads the file
{"action_type": "read_file", "path": "routes.py"}

# Step 2 — agent writes a minimal patch  
{"action_type": "write_patch", "file": "routes.py",
 "line_start": 74, "line_end": 74,
 "new_code": "    app.logger.info('User %s logged in', str(user.id))"}

# Step 3 — agent runs CI to verify
{"action_type": "run_ci"}
# CI returns: PASS | reward this step: +1.5 | deletion detected: No

# Step 4 — agent finalizes
{"action_type": "finalize_patch"}
# Final score: 1.7 / 2.0
```

| Action | Cost | What it does |
|--------|------|--------------|
| `read_file` | 1 from the per-episode file read budget | Returns file contents; counts against the read budget. |
| `write_patch` | Free (after reads) | Replaces a line range with new code. |
| `run_ci` | Free | Runs tests and static checks; returns step-level reward. |
| `finalize_patch` | Free | Ends the episode; final aggregate score. |

**File read budget** is what makes “read everything” impossible: on multi-file and microservice tasks, the agent must pick which file to open first. **Cross-file violations** are where shallow policies fail, because a fix in one file can break another, and the environment scores that.

## Reward Design

The hardest part was making the reward uncheateable.

| Signal | Value | When |
|--------|------:|------|
| CI passes | +1.0 | Violation pattern no longer detected. |
| No regressions | +0.5 | All files still parse and existing tests pass. |
| Patch is minimal | +0.2 | AST node delta under 3. |
| Unnecessary lines | -0.3 | Per extra line changed beyond the minimum needed. |
| Deletion detected | -1.0 | Removing the flagged line always scores -1.0, even if CI passes. |

**Anti-cheat, three mechanisms (by design):**

1. **Deletion penalty:** -1.0 is always on the table if the model tries to “fix” by deleting. You cannot outscore the rest of the signal when this fires.
2. **Hidden oracle:** catches hashing PII, shallow try/except silencing, and `TODO` comments that look like work but are not compliance repairs — the patterns that can slip past a shallow CI read.
3. **Five independent signals:** no one exploit can maximize all channels at once; a patch that is tiny can still fail hidden checks, and a patch that “passes” CI can still be scored down.

**Deletion** is the cheat most people try first, because it is the fastest edit. The table below is the one judges remember: it shows the same failure mode as a Python diff, in plain text.

```python
# What the agent tries (classic cheat):
# [line deleted]
# Reward: -1.0 — deletion detected regardless of CI result

# What the trained agent does instead:
app.logger.info('User %s logged in', str(user.id))
# Reward: +1.5 — full fix confirmed
```

## Tasks

| Task | Difficulty | Files | Violations | Frameworks |
|------|------------|------:|-----------:|------------|
| `task1_single_file` | Easy | 1 | 3 | Flask |
| `task2_django_app` | Medium | 5 | 8 | Django |
| `task2b_multifile_dependency` | Hard | 2 | 2 | Flask (cross-file) |
| `task3_microservices` | Hard | 7 | 15 | 4 microservices |
| `task4_django_rest` | Hard | 1 | 4 | Django REST |

## Training Results

**Model: Qwen2.5-3B-Instruct — Method: GRPO via TRL + Unsloth, 4-bit QLoRA, Colab T4 — Training: 120 real steps completed.**

| Metric | Before (base Qwen2.5-3B, zero GRPO) | After 120 GRPO steps |
|--------|-------------------------------------|------------------------|
| Valid JSON actions | ~50% of completions | ~83% at peak batches |
| Full fixes (reward > 1.0) | ~0% | 91% at peak batch |
| Deletion attempts | common | penalized and decreasing |

| Batch | Avg reward | Success rate | Note |
|------:|------------|--------------|------|
| 1 | +0.250 | 6/12 full fixes | |
| 5 | +0.508 | 5/12 full fixes | |
| 15 | +1.250 | 11/12 full fixes | ← peak |
| 19 | +1.083 | 10/12 full fixes | |

> **Note:** Batches 6-14 show collapse caused by a token truncation bug — 120 tokens was too short for `write_patch` JSON output. Fixed to 256 tokens mid-run. The recovery visible in batch 15 (+1.25) confirms the fix worked. This is what real RL training looks like.

![GRPO reward curve](reward_curve.png)

**Real-world relevance:** enterprise security teams doing SOC2 and GDPR certification still manually close hundreds of findings after static analysis — Semgrep, Bandit, and similar tools *find* issues; the human loop *fixes* them. An agent that scores above 0.8 on `task3_microservices` has learned behavior those teams can actually use, not a demo that only works on toy snippets.

## Self-Improving Adversary

An **adversary agent** generates new violations that are meant to be hard for the *current* patcher. The adversary earns reward only if the patcher **fails three consecutive attempts** on a generated case. That rule prevents reward hacking (random noise) and keeps pressure on the patcher to generalize, not memorize.

**Why the curriculum does not have to be hand-curated:** the adversary is trying to *win* only when the patcher truly fails repeatedly. As the patcher gets stronger, the winning adversarial patterns must move — otherwise they stop getting credit. The distribution shifts with capability instead of a fixed list of “hardest 10 files we could think of in week one.”

**Why this matters for OpenEnv-style demos:** a static benchmark can be overfit; a two-player loop is an honest stress test that grows with the policy being trained, without pretending that a single frozen task set is the whole story of compliance risk.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/skypank-coder/CompliancePatchBench
cd CompliancePatchBench && pip install -r requirements.txt
uvicorn api.server:app --port 7860
```

```bash
# 2. Run one episode
curl -X POST http://localhost:7860/patch/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task1_single_file"}'
```

```bash
# 3. Train your own agent
# Open in Colab → Runtime → Run All
# Link: https://github.com/skypank-coder/CompliancePatchBench/blob/main/project/colab_training.ipynb
```

## Project Structure

```text
CompliancePatchBench/
├── api/server.py
├── environment/patch_env.py
├── environment/hidden_compliance.py
├── environment/tasks/
├── project/rl_trainer.py
├── project/evaluate.py
├── project/agent.py
├── project/dataset_builder.py
├── project/colab_training.ipynb
├── Dockerfile
├── requirements.txt
└── project/requirements.txt
```

## Resources

- 🤗 [HF Space (live demo)](https://huggingface.co/spaces/skypank-coder/CompliancePatchBench)
- 📓 [Colab notebook](https://github.com/skypank-coder/CompliancePatchBench/blob/main/project/colab_training.ipynb) — [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skypank-coder/CompliancePatchBench/blob/main/project/colab_training.ipynb)
- 🔧 [GitHub repo](https://github.com/skypank-coder/CompliancePatchBench)
- 🧠 [HF adapter (trained weights)](https://huggingface.co/skypank-coder/compliancepatchbench-grpo-adapter)
- 📝 Blog post: [`BLOG.md`](BLOG.md) in repository root
- 🏆 Hackathon: Meta PyTorch OpenEnv Hackathon 2026, Bangalore Finals

*If your agent consistently scores above 0.8 on task3_microservices, it has learned something enterprise security teams would actually pay for.*
