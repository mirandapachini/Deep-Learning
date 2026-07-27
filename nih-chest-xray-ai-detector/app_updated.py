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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import zoom
import traceback
import uuid
import time

# Config
MODEL_PATH = "/Volumes/workspace/default/chest_xray_images/cxr14_inference_model.keras"
CLASSES_PATH = "cxr14_classes.json"
LAST_CONV_PATH = "cxr14_last_conv_layer.txt"
IMG_SIZE = (224, 224)
LOG_PATH = "cxr14_predictions_log.csv"
ERROR_LOG_PATH = "cxr14_error_log.csv"
USAGE_LOG_PATH = "cxr14_usage_log.csv"
PERFORMANCE_LOG_PATH = "cxr14_performance_log.csv"
FEEDBACK_LOG_PATH = "cxr14_feedback_log.csv"

# Logging Functions
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

# Model metrics from notebook analysis
per_condition_metrics = {
    "Atelectasis": {"AUC": 0.667, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
    "Cardiomegaly": {"AUC": 0.500, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
    "Consolidation": {"AUC": 0.500, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
    "Edema": {"AUC": 0.500, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
    "Effusion": {"AUC": 0.444, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
    "Emphysema": {"AUC": 0.500, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
    "Fibrosis": {"AUC": 0.500, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
    "Hernia": {"AUC": 0.500, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
    "Infiltration": {"AUC": 0.389, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
    "Mass": {"AUC": 0.500, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
    "Nodule": {"AUC": 0.667, "Precision": 0.500, "Recall": 0.500, "F1": 0.500},
    "Pleural_Thickening": {"AUC": 0.500, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
    "Pneumonia": {"AUC": 0.500, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
    "Pneumothorax": {"AUC": 0.500, "Precision": 0.000, "Recall": 0.000, "F1": 0.000},
}

# PAGE CONFIG
st.set_page_config(page_title="NIH ChestXray14 AI", page_icon="🫁", layout="wide")
st.title("🫁 NIH ChestXray14 Deep Learning Pipeline")
st.warning("⚠️ Educational/demo purposes only. Not a medical device or diagnostic tool.")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Analysis", "📊 Performance", "📈 Comparison", "🔬 EDA", "🧪 Test Set"])
