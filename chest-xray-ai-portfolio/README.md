Chest X-ray AI — Demo (local)
==============================

This folder contains the Streamlit demo for the Chest X-ray AI portfolio project.

Quick local run (recommended in a virtualenv):

1. Install dependencies:
```bash
cd chest-xray-ai-portfolio
pip install -r requirements.txt
```

2. Run the demo (keeps original `app.py` command working):
```bash
streamlit run app.py
```

Notes for reviewers
- The repository has been reorganized: the full app logic now lives in `src/chest_xray_ai_portfolio/app.py`.
- The top-level `chest-xray-ai-portfolio/app.py` is a small shim that sets `CHEST_XRAY_APP_DIR` and launches the relocated app so the demo can still be started with `streamlit run app.py`.
- If you prefer to run the app directly from `src/`, set the env var and run:
```bash
export CHEST_XRAY_APP_DIR=$(pwd)/chest-xray-ai-portfolio
python src/chest_xray_ai_portfolio/app.py
```

Data & model files
- Place `cxr14_inference_model.keras`, `cxr14_classes.json`, and `cxr14_last_conv_layer.txt` in the `chest-xray-ai-portfolio/data/` directory.

CI / Dev notes
- Consider adding a lightweight CI that installs `requirements.txt` and runs a smoke test such as importing the app module and verifying `load_model_and_classes()` falls back cleanly when model files are missing.
