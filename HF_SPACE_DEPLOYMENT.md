# Hugging Face Space Deployment

| Role | URL |
|------|-----|
| **API (FastAPI / Docker app)** | [huggingface.co/spaces/rachana05/Compliance-patch-bench](https://huggingface.co/spaces/rachana05/Compliance-patch-bench) |
| **Streamlit UI (Docker app)** | [huggingface.co/spaces/rachana05/CompliancePatchBench-UI](https://huggingface.co/spaces/rachana05/CompliancePatchBench-UI) |

**Canonical API Space (this monorepo’s Docker app):** `https://huggingface.co/spaces/rachana05/Compliance-patch-bench`

## Push API Space vs Streamlit UI Space (monorepo)

The **API** and **UI** are different Hugging Face Git repos. Pushing `main` from the monorepo root to **both** remotes is wrong for the UI: the repo’s **root** `Dockerfile` is the FastAPI server, so the Streamlit Space would build the backend image and never run Streamlit.

| Target | What to push | Command (from monorepo root) |
|--------|----------------|-----------------------------|
| **API** (`Compliance-patch-bench`) | Full repository: `Dockerfile`, `api/`, `environment/`, `project/`, … | `git push space main` |
| **UI** (`CompliancePatchBench-UI`) | **Only** `spaces/CompliancePatchBench-UI/` as the Space root (Streamlit `Dockerfile` + `src/`) | See below |

**Streamlit UI Space** (subtree — correct root `Dockerfile` for Streamlit):

```bash
git subtree split --prefix=spaces/CompliancePatchBench-UI -b hf-ui-space
git push ui hf-ui-space:main --force
```

Recreate `hf-ui-space` after UI changes so the split includes the latest commits. Use `--force` on the UI remote when replacing history; do **not** use the same full-repo push as the API.

The UI card `README.md` in that folder must keep `short_description` **≤ 60 characters** (Hub validation).

## `ENV_BASE_URL` (frontend → backend, required for a live demo)

The **Streamlit** Space is the **frontend**; the **FastAPI** Space is the **backend**. The UI’s Python code uses `ENV_BASE_URL` for all `requests` to `/health`, `/rl/learning-curve`, `/benchmark`, and patch endpoints. There is no client-side browser CORS to the API — the **Streamlit server** fetches the API.

**When demoing the hosted app**, in the **Streamlit** Space only ([CompliancePatchBench-UI](https://huggingface.co/spaces/rachana05/CompliancePatchBench-UI)) go to **Settings → Variables and secrets** and set:

- **`ENV_BASE_URL`** = the **deployed** API base URL, i.e. the public `https://…` of the **running** FastAPI Space (the `*.hf.space` URL you see in the address bar or Open in browser for the **API** Space, not the `huggingface.co/spaces/.../tree` page URL, and with **no** trailing path). Example shape: `https://rachana05-compliance-patch-bench.hf.space`

**Do not** use `http://localhost:7860` for a deployed Space: inside the UI container, `localhost` is the UI itself, not the API, so the frontend will not be connected to the backend and the app falls back to demo data.

**Local dev:** on your laptop, `http://127.0.0.1:7860` is fine if the API listens there; optional Docker: `http://host.docker.internal:7860` from a UI container to the host’s API.

**Check:** in a browser, `GET {ENV_BASE_URL}/health` should return `{"status":"ok",...}`. Then restart the UI Space and confirm the green “Environment Online” state (or the demo-data warning is gone).

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

## Why the Space says “No application file”

That message means the **Hugging Face Git repo is still empty** (or has no root `Dockerfile`). Building the image on your laptop is not enough — you must **push this repository** to the Space so the Hub can see `Dockerfile`, `app.py`, `api/`, etc.

## Push this repo to your Space (first time)

Use a [write token](https://huggingface.co/settings/tokens) (role **write**). If you use a different Space, substitute `rachana05/Compliance-patch-bench` below.

**Option A — Add a remote to your existing clone**

If you already have a `space` remote pointing elsewhere (e.g. another account), update it:

```bash
git remote set-url space https://huggingface.co/spaces/rachana05/Compliance-patch-bench
```

Otherwise add it once:

```bash
cd /path/to/CompliancePatchBench
git remote add space https://huggingface.co/spaces/rachana05/Compliance-patch-bench
git push -u space main
```

If the Space already has an auto-generated first commit, either pull and merge first, or (only if you intend to replace it entirely with your project):

```bash
git push -u space main:main --force
```

**Option B — Clone the empty Space, copy files, push**

```bash
git clone https://huggingface.co/spaces/rachana05/Compliance-patch-bench
cd Compliance-patch-bench
# copy your project files into this folder (Dockerfile, app.py, api/, environment/, project/, requirements.txt, README.md, .dockerignore, …)
git add -A
git status   # you should see Dockerfile at repo root
git commit -m "Add CompliancePatchBench Docker app"
git push
```

**Auth:** if Git asks for a password, use your Hugging Face **token**, not your account password. Or run `huggingface-cli login` and use a credential helper.

### `Invalid username or password` / `Authentication failed` on `git push`

1. **Password must be a token, not your login password**  
   Create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **Write** (or a fine-grained token that can **write** to that Space).

2. **Username is your Hub username (e.g. `rachana05`)**, not your email. If Git only prompts for “password”, the username is often taken from the URL or a prior login — if macOS Keychain stored the wrong pair, the next point fixes that.

3. **Clear a bad saved password (macOS)**  
   In **Keychain Access**, search for `huggingface` or `huggingface.co`, delete the **internet password** entry used for `git`, then push again and enter **username** + **token** when prompted.  
   Or run once:  
   `printf "host=huggingface.co\nprotocol=https\n" | git credential-osxkeychain erase`

4. **One-line check without the keychain** (token visible in process list; prefer clearing keychain in daily use):
   `git push https://rachana05:YOUR_HF_TOKEN@huggingface.co/spaces/rachana05/Compliance-patch-bench main`

5. **SSH (optional):** [Add an SSH key](https://huggingface.co/docs/huggingface_hub/main/guides/credentials) to your HF account, then:
   `git remote set-url space git@hf.co:spaces/rachana05/Compliance-patch-bench`  
   and `git push -u space main`. No password prompt if your ssh-agent has the right key.

After a successful push, open the Space **“App”** tab: status should go from “No application file” to **Building**, then **Running** (this can take several minutes on first build). Watch **Logs** for Uvicorn on port `7860` and for `GET /health` `200 OK`.

## Hugging Face Space Setup (checklist)

1. Create a new Space.
2. Select **Docker** as the SDK.
3. **Push** this repository (steps above) — the yellow “No application file” bar disappears once `Dockerfile` exists on the `main` branch.
4. Confirm the Space logs show Uvicorn serving on the configured `PORT`.
5. Open `https://YOUR_USERNAME-YOUR_SPACE.hf.space/health` (or your Space URL) — use the Space URL, not `0.0.0.0` in the browser.
6. For local review: `/health`, `/project`, `/rl/learning-curve`, `/tasks`, or `/benchmark`.

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
hf download rachana05/Compliance-patch-bench --repo-type=space --local-dir ./hf-space-download
```

Alternative download path:

```bash
git lfs install
git clone https://huggingface.co/spaces/rachana05/Compliance-patch-bench
```

## Reviewer Endpoints

- `/project` explains the RL formulation, anti-cheat reward, and latest learning-curve point.
- `/rl/learning-curve` exposes reward, success, and hidden-violation history by iteration.
- `/benchmark` exposes task difficulty and confirms hidden constraints / GRPO policy optimization support.

Before pushing, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m project.smoke_test
```
