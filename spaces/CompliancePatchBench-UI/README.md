---
title: CompliancePatchBench UI
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
short_description: Streamlit demo; set ENV_BASE_URL to your API Space
license: mit
pinned: false
---

# CompliancePatchBench — Demo UI

[Main repository](https://github.com/skypank-coder/CompliancePatchBench)

**Required Space variable:** In **Settings → Variables and secrets**, add:

- `ENV_BASE_URL` = the base URL of your **FastAPI** CompliancePatchBench Space (e.g. `https://<user>-<api-space-name>.hf.space` — no trailing path).  
  The UI calls `/health`, `/rl/learning-curve`, `/benchmark`, `/patch/reset`, `/patch/step`.

Optional:

- `CPB_GITHUB_URL` — GitHub link for the sidebar  
- `CPB_HF_SPACE_URL` — link shown as “Hugging Face Space” (use your **API** Space or main repo)

Without `ENV_BASE_URL`, the app uses demo data and shows a warning.
