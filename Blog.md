# We trained an LLM to fix security violations — and made sure it couldn't cheat

*Training a compliance patch agent with GRPO, a deletion-proof reward function, and a self-improving adversary.*

You ask a vanilla LLM to fix a GDPR logging violation. It "fixes" it by deleting the line, or by hashing the PII so the string looks clean, or by dropping a TODO comment. Your CI may still pass. None of that survives a real audit. CompliancePatchBench is built to train an agent that actually repairs the code. How do you reward that without letting the model game the score?

## The problem with asking an LLM to fix your compliance violations

Example one: the model deletes the offending line. The violation vanishes; the app breaks. Example two: it hashes the email before logging. The rule is still wrong; you are still processing identifying data carelessly, just with a different shape. Example three: it leaves `# TODO: fix GDPR` and moves on. A real fix removes or replaces the behavior so policy and code match. Most off-the-shelf LLM patches do not do that, even when the visible linter goes quiet.

## The environment

Each episode is a small game. The agent starts with a Python codebase, a list of flagged violations (with rule ids and line ranges), and a **read budget** (it cannot read every file, so on harder tasks it has to choose what to open). It may **read_file**, **write_patch** (replace a line range with new code), **run_ci** (re-check against the semantic checker), or **finalize_patch** to end the episode. The episode ends on finalize, or when step limits and budgets are exhausted.

```text
# Agent receives:
# - routes.py (180 lines, Flask app)
# - Violation: GDPR-ART5-1A at line 74
# - Read budget: 3 files

# Agent's first action:
{"action_type": "read_file", "path": "routes.py"}

# Agent's second action (after reading):
{"action_type": "write_patch", "file": "routes.py", "line_start": 74, "line_end": 74,
 "new_code": "    app.logger.info('User %s logged in', str(user.id))"}

# Agent's third action:
{"action_type": "run_ci"}

# CI returns: PASS | reward: +1.5
```

The agent that plays this game well is one that has learned to read, patch minimally, and verify, not to bluff.

## The reward function that cannot be gamed

The hardest part of this project was not the FastAPI `reset` / `step` shell. It was the reward. The most obvious exploit is **deletion**: strip the line, the pattern disappears, and a naive "CI only" score would look great. In CompliancePatchBench, **if the agent removes the flagged line without a real replacement, the deletion signal always gives -1.0**, even when something upstream still prints PASS. You cannot "accidentally" max reward by erasing the problem.

The next exploit is subtler: **hash PII, wrap the log in `try/except`, or add a comment instead of a fix**. A hidden oracle looks for these shortcuts. They can pass a shallow check; they do not pass the second pass. The five reward channels are **independent** (+1.0 CI, +0.5 no regressions, +0.2 minimal AST delta, -0.3 per unnecessary line, -1.0 deletion). There is no single dial that spams a win without doing real work.

## Training: what GRPO on a real environment looks like

We trained **Qwen2.5-3B-Instruct** with **GRPO** in TRL on top of **Unsloth 4-bit QLoRA** on a **Colab T4** for **120** optimizer steps. The curve is not a smooth hockey stick. Real batch means: batch 1 mean reward **+0.250**, 6/12 success; batch 5 **+0.508**, 5/12; batch 15 **+1.250**, 11/12 success (best); batch 19 **+1.083**, 10/12. Batches 6-14 **collapsed toward -1.0** because **completion length was truncated at 120 tokens**, which is too short for valid `write_patch` JSON. After raising the cap to **256** tokens, batches settled in the +1.0 to +1.25 band. That ugly middle is what honest RL on a real env looks like when your infra has a bug.

| | Base Qwen2.5-3B | After 120 GRPO steps |
|---|---|---|
| Valid JSON rate | ~50% | ~83% in peak batches |
| Full fix rate (reward > 1.0) | ~0% | 91% in the best batch |
| Deletion-style shortcuts | Common | Penalized, decreasing |

**Bad patch (still "passes" shallow checks, fails our reward):**  
`# logging removed for compliance` → deletion detected, **-1.0**.

**Good patch (from training):**  
Violation: `app.logger.info(f"User {user.email} logged in")`  
Patch: `app.logger.info("User %s logged in", str(user.id))` → **CI PASS, +1.5**, deletion: no.

## The self-improving loop

After the patcher trains, a second **adversary** agent writes new violations meant to break the patcher. It only earns reward if the patcher **fails three times** in a row. The curriculum **scales with** the patcher, so we are not hand-curating an endless list of "hard" tasks. The two agents push each other.

## Who should care

**If you run security or compliance for a SaaS** and you already ship Semgrep or similar, you know the gap: finders are cheap, human remediation is the bottleneck. An agent that scores real fixes on multi-file and cross-service tasks is the difference between a slide deck and a product.

**If you study reward hacking**, this is a concrete case where the dominant exploit (delete the line) is **shut off** by a dedicated term that cannot be bypassed for a quick win, and where a second pass catches semantic cheating that greenlights in CI.

**If you build OpenEnv-style tasks**, the lesson is that **anti-cheat is part of the environment design**: independent signals, explicit deletion handling, and an oracle for shortcuts that "look" fixed.

## Try it

- **Live API (Hugging Face Space):** [huggingface.co/spaces/rachana05/Compliance-patch-bench](https://huggingface.co/spaces/rachana05/Compliance-patch-bench)
- **Training notebook (GitHub; Open in Colab from the file menu):** [github.com/skypank-coder/CompliancePatchBench/blob/main/project/model_training_.ipynb](https://github.com/skypank-coder/CompliancePatchBench/blob/main/project/model_training_.ipynb)
- **Public Colab (optional):** [Open in Colab](https://colab.research.google.com/drive/1d-rzhyYXo6LrHsMV924lUcNv3663cs-o?usp=sharing) (same project; `openenv.yaml` stores this under `links.colab_notebook`)
- **Code and data:** [github.com/skypank-coder/CompliancePatchBench](https://github.com/skypank-coder/CompliancePatchBench)
- **LoRA adapter:** [huggingface.co/skypank-coder/compliancepatchbench-grpo-adapter](https://huggingface.co/skypank-coder/compliancepatchbench-grpo-adapter)

If your agent can consistently score above 0.8 on task3_microservices, it has learned something that enterprise security teams would actually pay for.
