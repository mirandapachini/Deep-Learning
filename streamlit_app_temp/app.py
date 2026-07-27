
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import json
import pandas as pd
import datetime
import os
import matplotlib.cm as cm
from scipy.ndimage import zoom
import traceback
import uuid
import time


# Config - UPDATED FOR DATABRICKS APPS
# Model stored in Unity Catalog Volume (accessible to app service principal)
MODEL_PATH = "/Volumes/workspace/default/chest_xray_images/cxr14_inference_model.keras"
CLASSES_PATH = "cxr14_classes.json"  # Local to app directory
LAST_CONV_PATH = "cxr14_last_conv_layer.txt"  # Local to app directory
IMG_SIZE = (224, 224)
LOG_PATH = "cxr14_predictions_log.csv"
ERROR_LOG_PATH = "cxr14_error_log.csv"
USAGE_LOG_PATH = "cxr14_usage_log.csv"
PERFORMANCE_LOG_PATH = "cxr14_performance_log.csv"
FEEDBACK_LOG_PATH = "cxr14_feedback_log.csv"

# NOTE: Model must be in UC Volume with READ VOLUME permission granted to app service principal
# Grant permission: GRANT READ VOLUME ON VOLUME workspace.default.chest_xray_images TO `<service-principal-id>`


# Advanced Logging Functions
def log_error(error_type, error_message, context=""):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_data = {"timestamp": timestamp, "error_type": error_type, "error_message": str(error_message), "context": context, "traceback": traceback.format_exc()}
        df = pd.DataFrame([error_data])
        df.to_csv(ERROR_LOG_PATH, mode='a', header=not os.path.exists(ERROR_LOG_PATH), index=False)
    except: pass

def log_usage(action, details="", session_id=None):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        usage_data = {"timestamp": timestamp, "session_id": session_id or "unknown", "action": action, "details": details}
        df = pd.DataFrame([usage_data])
        df.to_csv(USAGE_LOG_PATH, mode='a', header=not os.path.exists(USAGE_LOG_PATH), index=False)
    except: pass

def log_performance(operation, duration, session_id=None, context=""):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        perf_data = {"timestamp": timestamp, "session_id": session_id or "unknown", "operation": operation, "duration_seconds": round(duration, 3), "context": context}
        df = pd.DataFrame([perf_data])
        df.to_csv(PERFORMANCE_LOG_PATH, mode='a', header=not os.path.exists(PERFORMANCE_LOG_PATH), index=False)
    except: pass

def log_feedback(feedback_text, context, session_id=None):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feedback_data = {"timestamp": timestamp, "session_id": session_id or "unknown", "feedback": feedback_text, "context": context}
        df = pd.DataFrame([feedback_data])
        df.to_csv(FEEDBACK_LOG_PATH, mode='a', header=not os.path.exists(FEEDBACK_LOG_PATH), index=False)
    except: pass

def log_prediction(image_name, predictions, threshold, colormap, session_id=None):
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        max_prob, min_prob, mean_prob = float(np.max(predictions)), float(np.min(predictions)), float(np.mean(predictions))
        num_above_50, num_above_threshold = int(np.sum(predictions > 0.5)), int(np.sum(predictions >= threshold))
        top_idx = np.argmax(predictions)
        row = {"timestamp": timestamp, "session_id": session_id or "unknown", "image_name": image_name, "threshold_used": threshold, "colormap_used": colormap, "top_prediction": CLASSES[top_idx], "top_probability": float(predictions[top_idx]), "max_prob": max_prob, "min_prob": min_prob, "mean_prob": mean_prob, "num_above_50_percent": num_above_50, "num_above_threshold": num_above_threshold}
        for i, cls in enumerate(CLASSES): row[cls] = predictions[i]
        df = pd.DataFrame([row])
        df.to_csv(LOG_PATH, mode='a', header=not os.path.exists(LOG_PATH), index=False)
    except Exception as e: log_error("PredictionLoggingError", str(e), "log_prediction function")


@st.cache_resource
def load_model_and_classes():
    try:
        model = keras.models.load_model(MODEL_PATH)
        with open(CLASSES_PATH, "r") as f: classes = json.load(f)
        last_conv = None
        if os.path.exists(LAST_CONV_PATH):
            with open(LAST_CONV_PATH, "r") as f: last_conv = f.read().strip()
        log_usage("model_loaded", f"Model loaded from {MODEL_PATH}", session_id="system")
        return model, classes, last_conv
    except Exception as e:
        log_error("ModelLoadError", str(e), "Failed to load model on startup")
        raise

model, CLASSES, LAST_CONV_LAYER = load_model_and_classes()


def make_gradcam_heatmap(img_array, class_index, model, last_conv_layer_name):
    base_model = model.layers[0]
    grad_model = tf.keras.models.Model([model.inputs], [base_model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image, colormap_name="hot", alpha=0.4):
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    zoom_factors = (image.size[1] / heatmap.shape[0], image.size[0] / heatmap.shape[1])
    heatmap_resized = zoom(heatmap, zoom_factors, order=1)
    if colormap_name == "red":
        heatmap_colored = np.zeros((*heatmap_resized.shape, 3), dtype=np.uint8)
        heatmap_colored[:, :, 0] = np.uint8(255 * heatmap_resized)
    else:
        cmap = cm.get_cmap('hot' if colormap_name == 'hot' else 'viridis')
        heatmap_colored = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    return Image.blend(image.convert("RGB"), Image.fromarray(heatmap_colored), alpha=alpha)


condition_definitions = {"Atelectasis": "Partial collapse of part of the lung.", "Cardiomegaly": "Enlarged heart silhouette.", "Effusion": "Fluid between lung and chest wall.", "Infiltration": "Hazy opacity from infection or inflammation.", "Mass": "Large, well-defined abnormal lung area.", "Nodule": "Small round spot in the lung.", "Pneumonia": "Infection causing lung opacities.", "Pneumothorax": "Air between lung and chest wall.", "Consolidation": "Dense opacity where air is replaced by fluid.", "Edema": "Fluid accumulation inside lung tissue.", "Emphysema": "Over-inflated, damaged air sacs.", "Fibrosis": "Scarring/thickening of lung tissue.", "Pleural_Thickening": "Thickened lining around the lung.", "Hernia": "Organ protruding into chest cavity."}


st.title("NIH ChestXray14 Deep Learning Pipeline")
st.warning("⚠️ This tool is for educational/demo purposes only and is not a medical device or diagnostic tool. Not intended for clinical use.")

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
    st.session_state.error_count = 0
    log_usage("app_started", "User opened application", session_id=st.session_state.session_id)

session_id = st.session_state.session_id

uploaded_file = st.file_uploader("Drag and drop a chest X-ray image here, or click to browse.", type=["png", "jpg", "jpeg"])

if uploaded_file and 'last_uploaded' not in st.session_state:
    st.session_state.last_uploaded = uploaded_file.name
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    log_usage("image_uploaded", f"filename={uploaded_file.name}, size={file_size_mb:.2f}MB", session_id=session_id)
elif uploaded_file and st.session_state.get('last_uploaded') != uploaded_file.name:
    st.session_state.last_uploaded = uploaded_file.name
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    log_usage("image_uploaded", f"filename={uploaded_file.name}, size={file_size_mb:.2f}MB", session_id=session_id)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("🔍 Analyze Image", type="primary"):
        log_usage("analyze_clicked", f"image={uploaded_file.name}", session_id=session_id)
        with st.spinner("Analyzing..."):
            try:
                start_preprocess = time.time()
                x = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
                x = np.expand_dims(np.array(x) / 255.0, 0)
                log_performance("preprocessing", time.time() - start_preprocess, session_id=session_id, context=uploaded_file.name)
                
                start_pred = time.time()
                preds = model.predict(x, verbose=0)[0]
                log_performance("prediction", time.time() - start_pred, session_id=session_id, context=uploaded_file.name)
                
                log_usage("prediction_generated", f"image={uploaded_file.name}, top={CLASSES[np.argmax(preds)]}", session_id=session_id)
                if st.session_state.error_count > 0:
                    log_usage("recovered_from_error", f"after={st.session_state.error_count}_errors", session_id=session_id)
                    st.session_state.error_count = 0
            except Exception as e:
                st.session_state.error_count += 1
                log_error("PredictionError", str(e), f"Failed to analyze {uploaded_file.name}")
                log_usage("error_occurred", f"error_count={st.session_state.error_count}", session_id=session_id)
                st.error(f"⚠️ Error analyzing image: {str(e)}")
                st.stop()

            col_thresh, col_cmap = st.columns(2)
            with col_thresh:
                threshold = st.slider("Probability threshold for highlighting findings", 0.0, 1.0, 0.5, 0.05, key="threshold_slider")
                if 'last_threshold' not in st.session_state: st.session_state.last_threshold = threshold
                elif st.session_state.last_threshold != threshold:
                    log_usage("threshold_changed", f"from={st.session_state.last_threshold} to={threshold}", session_id=session_id)
                    st.session_state.last_threshold = threshold
            with col_cmap:
                colormap_choice = st.selectbox("Grad-CAM Colormap (for accessibility)", ["hot", "red", "viridis"], 0, help="Hot: Colorblind-friendly\nRed: Original\nViridis: Perceptually uniform", key="colormap_selector")
                if 'last_colormap' not in st.session_state: st.session_state.last_colormap = colormap_choice
                elif st.session_state.last_colormap != colormap_choice:
                    log_usage("colormap_changed", f"from={st.session_state.last_colormap} to={colormap_choice}", session_id=session_id)
                    st.session_state.last_colormap = colormap_choice

            log_prediction(uploaded_file.name, preds, threshold, colormap_choice, session_id=session_id)
            st.markdown("---")

            findings = sorted(list(zip(CLASSES, preds)), key=lambda x: x[1], reverse=True)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("All Predicted Findings")
                for cls, p in findings:
                    if p >= threshold: st.markdown(f"**[HIGH] {cls}** - probability: `{p:.3f}`")
                    else: st.write(f"**{cls}** - probability: `{p:.3f}`")
                    if cls in condition_definitions: st.caption(condition_definitions[cls])
            with col2:
                st.subheader("Top 5 Findings")
                for cls, p in findings[:5]: st.metric(label=cls, value=f"{p:.1%}")

            st.markdown("---")
            st.subheader(f"Findings Above Threshold ({threshold:.2f})")
            positives = [(cls, p) for cls, p in findings if p >= threshold]
            if positives:
                for cls, p in positives:
                    st.success(f"✅ **{cls}** - `{p:.3f}`")
                    with st.expander(f"View Grad-CAM Heatmap for {cls}"):
                        log_usage("gradcam_viewed", f"condition={cls}, colormap={colormap_choice}", session_id=session_id)
                        try:
                            start_gc = time.time()
                            heatmap = make_gradcam_heatmap(x, CLASSES.index(cls), model, LAST_CONV_LAYER)
                            overlay_img = overlay_heatmap(heatmap, image, colormap_name=colormap_choice)
                            log_performance("gradcam", time.time() - start_gc, session_id=session_id, context=f"{cls}_{colormap_choice}")
                            st.image(overlay_img, caption=f"Grad-CAM: {cls} (Colormap: {colormap_choice})", use_column_width=True)
                            if colormap_choice == "hot": st.caption("🔥 Hot areas (red→yellow→white) show focus. Colorblind-friendly.")
                            elif colormap_choice == "red": st.caption("🔴 Red areas show focus. Brighter = higher importance.")
                            else: st.caption("💜 Bright areas (purple→green→yellow) show focus. Perceptually uniform.")
                        except Exception as e:
                            log_error("GradCAMError", str(e), f"Failed for {cls}")
                            st.error(f"⚠️ Error generating heatmap: {str(e)}")
            else:
                st.info("No findings above threshold.")

            st.markdown("---")
            st.subheader("Power BI Integration")
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "rb") as f:
                    st.download_button("⬇️ Download Predictions Log (CSV)", f, "cxr14_predictions_log.csv", "text/csv", on_click=lambda: log_usage("download_predictions", "User downloaded CSV", session_id=session_id))
                st.caption("Import this CSV into Power BI for trend analysis and dashboards.")
            
            st.markdown("---")
            with st.expander("📝 Report an Issue or Provide Feedback"):
                st.info("Having trouble? Let us know! Your feedback helps improve the app.")
                feedback_text = st.text_area("Describe the issue or share your feedback:", placeholder="Example: The Grad-CAM heatmap didn't load...", key="feedback_input")
                if st.button("Submit Feedback", type="secondary"):
                    if feedback_text.strip():
                        log_feedback(feedback_text, f"image={uploaded_file.name}, threshold={threshold}, colormap={colormap_choice}", session_id=session_id)
                        log_usage("feedback_submitted", f"length={len(feedback_text)}", session_id=session_id)
                        st.success("✅ Feedback submitted! Thank you.")
                    else:
                        st.warning("⚠️ Please enter feedback before submitting.")
            
            with st.expander("📊 Developer Logs (Usage, Performance & Errors)"):
                st.info("Logs track usage, performance, and errors.")
                st.caption(f"**Session ID:** `{session_id}`")
                col_u, col_p, col_e = st.columns(3)
                with col_u:
                    st.markdown("**Usage**")
                    if os.path.exists(USAGE_LOG_PATH):
                        with open(USAGE_LOG_PATH, "rb") as f: st.download_button("⬇️ Usage", f, "cxr14_usage_log.csv", "text/csv", key="dl_usage")
                        usage_df = pd.read_csv(USAGE_LOG_PATH)
                        st.caption(f"Total: {len(usage_df)}")
                        if 'session_id' in usage_df.columns: st.caption(f"Sessions: {usage_df['session_id'].nunique()}")
                    else: st.caption("No data yet.")
                with col_p:
                    st.markdown("**Performance**")
                    if os.path.exists(PERFORMANCE_LOG_PATH):
                        with open(PERFORMANCE_LOG_PATH, "rb") as f: st.download_button("⬇️ Perf", f, "cxr14_performance_log.csv", "text/csv", key="dl_perf")
                        perf_df = pd.read_csv(PERFORMANCE_LOG_PATH)
                        if len(perf_df) > 0:
                            avg_pred = perf_df[perf_df['operation']=='prediction']['duration_seconds'].mean()
                            st.caption(f"Avg pred: {avg_pred:.2f}s")
                    else: st.caption("No data yet.")
                with col_e:
                    st.markdown("**Errors**")
                    if os.path.exists(ERROR_LOG_PATH):
                        with open(ERROR_LOG_PATH, "rb") as f: st.download_button("⬇️ Errors", f, "cxr14_error_log.csv", "text/csv", key="dl_errors")
                        error_df = pd.read_csv(ERROR_LOG_PATH)
                        st.caption(f"⚠️ Total: {len(error_df)}")
                    else: st.caption("✅ No errors!")
                
                st.markdown("---")
                st.markdown("**User Feedback**")
                if os.path.exists(FEEDBACK_LOG_PATH):
                    col_fb1, col_fb2 = st.columns([1, 3])
                    with col_fb1:
                        with open(FEEDBACK_LOG_PATH, "rb") as f: st.download_button("⬇️ Feedback", f, "cxr14_feedback_log.csv", "text/csv", key="dl_feedback")
                    with col_fb2:
                        feedback_df = pd.read_csv(FEEDBACK_LOG_PATH)
                        st.caption(f"💬 {len(feedback_df)} submissions")
                    if len(feedback_df) > 0:
                        st.dataframe(feedback_df[["timestamp", "session_id", "feedback", "context"]].tail(5), use_container_width=True)
                else: st.caption("No feedback yet.")
