## Chest X-ray AI — Employer Portfolio

This repository holds a set of deep-learning projects and artifacts focused on medical imaging, with a polished employer-facing demo for multi-label chest X-ray classification.

Highlights
- Streamlit demo showcasing Grad-CAM interpretability and multi-label predictions for 14 chest conditions
- Keras/TensorFlow inference pipeline with reproducible preprocessing and logging
- Notebooks and documentation demonstrating model training, evaluation, and deployment guidance

If you're reviewing this repository as a hiring manager or collaborator, start with the `chest-xray-ai-portfolio/` folder for the interactive demo and `notebooks/` for supporting analysis.

Quick Start (local)
1. Install Python 3.8+ and virtualenv/venv
2. From repository root, install the demo dependencies:
```
cd chest-xray-ai-portfolio
pip install -r requirements.txt
```
3. Run the Streamlit demo:
```
streamlit run app.py
```

What I recommend reviewing for interviews
- `chest-xray-ai-portfolio/app.py`: production-style Streamlit demo with inference and Grad-CAM
- `notebooks/`: training and evaluation notebooks (narrative + results)
- `chest-xray-ai-portfolio/docs/`: project-specific docs (threshold tuning, Grad-CAM notes)

Contact
- Miranda Pachini — see GitHub profile for contact and portfolio links

More details and full project documentation have been archived to `docs/README_ARCHIVE.md`.