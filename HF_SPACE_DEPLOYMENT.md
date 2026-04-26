# Hugging Face Space Deployment (API only)

| Role | URL (example) |
|------|----------------|
| **API (FastAPI / Docker app)** | [huggingface.co/spaces/rachana05/Compliance-patch-bench](https://huggingface.co/spaces/rachana05/Compliance-patch-bench) |

**Canonical base:** push this monorepo **`main`** to the Space that hosts the **root `Dockerfile`** (FastAPI).

## Push the API Space (from monorepo root)

Use a [write token](https://huggingface.co/settings/tokens). If the remote is named `space`:

```bash
git remote add space https://huggingface.co/spaces/OWNER/SPACE-NAME
git push -u space main
```

If the Space already has an unrelated first commit, you may need `git push -u space main:main --force` (only if you intend to replace it).

## `ENV` / training data

- The API serves `project/data/learning_curve.json` (see `GET /training-curve`, `GET /rl/learning-curve`).
- **Optional upload:** `POST /upload-training-log` with `CPB_TRAINING_UPLOAD_TOKEN` and header `X-CPB-Token` (see `api/server.py`).

## Public Colab → API

- Training writes files on your machine or Colab. To update what the **deployed** API returns, add updated JSON under `project/data/`, commit, and **redeploy** the Space, or use the upload endpoint when enabled.

## Local smoke test

```bash
docker build -t compliancepatchbench .
docker run --rm -p 7860:7860 compliancepatchbench
curl http://localhost:7860/health
```

## Reviewer endpoints

- `/health`, `/project`, `/tasks`, `/benchmark`, `/training-curve`, `/rl/learning-curve`, `/patch/reset`, `/patch/step`, …

Run `PYTHONDONTWRITEBYTECODE=1 python -m project.smoke_test` before pushing when you change training/API wiring.
