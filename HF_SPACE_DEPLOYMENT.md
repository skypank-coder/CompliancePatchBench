# Hugging Face Space Deployment

This repo is ready to deploy as a Docker-based Hugging Face Space.

## Runtime

- SDK: Docker
- Port: `${PORT:-7860}`
- Entrypoint: `uvicorn app:app --host 0.0.0.0 --port 7860`
- Health check: `GET /health`

## Files Required In The Docker Image

The runtime image needs:

- `api/`
- `environment/`
- `project/`
- `requirements.txt`
- `Dockerfile`

Generated notebooks, bytecode caches, local evaluation logs, LoRA adapters, and
large training artifacts are excluded by `.dockerignore` so the Space stays
small and stages quickly. The small demo RL artifacts
`project/data/learning_curve.json` and `project/data/tabular_rl_policy.json`
are intentionally included so reviewer endpoints show useful data in production.

## Local Smoke Test

```bash
docker build -t compliancepatchbench .
docker run --rm -p 7860:7860 compliancepatchbench
curl http://localhost:7860/health
```

Expected response:

```json
{"status":"ok","version":"1.0.0"}
```

## Hugging Face Space Setup

1. Create a new Space.
2. Select **Docker** as the SDK.
3. Push this repository.
4. Confirm the Space logs show Uvicorn serving on the configured `PORT`.
5. Open `/health`, `/project`, `/rl/learning-curve`, `/tasks`, or `/benchmark`.

The training notebooks and RL demo remain available in the repository for
judging, but they are not copied into the Docker image.

## Hugging Face CLI Troubleshooting

If `hf download ... --repo-type=space` fails with:

```text
TypeError: Typer.__init__() got an unexpected keyword argument 'suggest_commands'
```

your local Hugging Face CLI has a dependency mismatch. Upgrade the CLI stack:

```bash
python3 -m pip install --user --upgrade huggingface_hub "typer>=0.24.2" "click>=8.2.1"
hash -r
hf --version
```

Then retry:

```bash
hf download Skypank/compliance-patch-bench --repo-type=space --local-dir ./hf-space-download
```

Alternative download path:

```bash
git lfs install
git clone https://huggingface.co/spaces/Skypank/compliance-patch-bench
```

## Reviewer Endpoints

- `/project` explains the RL formulation, anti-cheat reward, and latest learning-curve point.
- `/rl/learning-curve` exposes reward, success, and hidden-violation history by iteration.
- `/benchmark` exposes task difficulty and confirms hidden constraints / GRPO policy optimization support.

Before pushing, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m project.smoke_test
```
