Short (hiring-focused)
----------------------
I'm hiring-focused and open to roles that let me ship ML demos and production-ready tooling. I just finished a focused reorg of my Chest X‑ray AI portfolio to make it resume- and interview-ready:

- Clear source layout: moved the Streamlit demo into `src/chest_xray_ai_portfolio/app.py` and added a compatibility shim at `chest-xray-ai-portfolio/app.py`.
- Reproducibility & CI: pinned dependencies in `chest-xray-ai-portfolio/requirements.txt` and added a GitHub Actions smoke-check to catch regressions early.
- Project hygiene: added an MIT `LICENSE`, `CONTRIBUTING.md`, and a concise, recruiter-friendly `README.md`.

Model files are excluded for size/privacy — to run the demo locally, add the model and support files under `chest-xray-ai-portfolio/data/`.

If you're hiring for ML engineering, MLOps, or applied research roles and want someone who can move prototypes to demo-ready products, please DM me.

---

Long (hiring-focused post)
--------------------------
I reorganized and hardened my Chest X‑ray AI portfolio to be easy-to-review and production-aware — the kind of repo I want to present to hiring managers.

Key outcomes:
- Source & runability: the demo is now under `src/` (Streamlit app) with a lightweight shim to preserve `streamlit run app.py` compatibility — this reduces onboarding time for reviewers.
- Reproducible dev flow: dependency versions are pinned and a CI smoke-check runs simple import/compile checks to prevent trivial PR regressions.
- Professional packaging: added `LICENSE` (MIT), `CONTRIBUTING.md`, and a focused top-level `README.md` so reviewers see goals, how to run, and evaluation notes immediately.

Impact for hiring reviews:
- Shortens the evaluation loop — reviewers can quickly run the demo or scan the code to assess engineering practices.
- Demonstrates end-to-end skills: model inference + visualization (Grad-CAM), reproducible environment, CI, and clear documentation.

Next steps I can help with for hiring rounds:
- Create a short screencast demo or GIF showing the app in action.
- Produce a single-page README summary tuned for recruiters (skills + impact + time-to-run).
- Add a CI mock-inference job so CI validates higher-level behavior without heavy TF installs.

If you're hiring for ML engineering, MLOps, or applied research, I'd welcome a conversation — DM me to schedule a quick walkthrough.

#MachineLearning #MLOps #AppliedML #Streamlit #Hiring
