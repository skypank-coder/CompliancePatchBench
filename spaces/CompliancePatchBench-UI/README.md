---
title: CompliancePatchBench UI
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
short_description: "Streamlit UI — set ENV_BASE_URL to your API *.hf.space URL"
license: mit
pinned: false
---

# CompliancePatchBench — Demo UI

[Main repository](https://github.com/skypank-coder/CompliancePatchBench)

**Required for a real demo (frontend connected to backend):** in **Settings → Variables and secrets** on this **Streamlit** Space, set:

- **`ENV_BASE_URL`** = the **deployed** base URL of your **FastAPI** Space — the public `https://…` that serves the live API, typically `https://<user>-<api-space>.hf.space` (copy from the API Space’s **Open** / in-browser app URL, **not** the `huggingface.co/spaces/...` repo page, **not** a path like `/docs`). **No** trailing slash.  
- **Not** `http://localhost:7860` when this UI runs on Hugging Face: the UI container’s `localhost` is not the API Space, so the Streamlit app would stay on demo data.  
- The UI (server-side `requests`) calls: `/health`, `/rl/learning-curve`, `/benchmark`, `/patch/reset`, `/patch/step`.  
- **Restart** the Space after changes. Test: `curl` or open `{ENV_BASE_URL}/health` — expect `{"status":"ok",...}`.

Optional:

- `CPB_GITHUB_URL` — GitHub link in the sidebar and footer  
- `CPB_HF_SPACE_URL` — **API** Space page URL (default: `https://huggingface.co/spaces/rachana05/Compliance-patch-bench`)  
- `CPB_HF_UI_SPACE_URL` — **Streamlit** Space page URL (default: `https://huggingface.co/spaces/rachana05/CompliancePatchBench-UI`)

Without `ENV_BASE_URL`, the app uses demo data and shows a warning.
