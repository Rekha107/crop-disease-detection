import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import streamlit as st
import pandas as pd
from datetime import datetime
from fpdf import FPDF
import tempfile
import base64

MODEL_PATH = "best_model.h5"
NAMES_PATH = "class_names.json"
IMG_SIZE   = (224, 224)

DISEASE_DB = {
    "Apple___Apple_scab": {
        "severity": "Medium", "cause": "Fungal (Venturia inaequalis)",
        "symptoms": "Olive-green or brown spots on leaves and fruit",
        "treatment": "Apply fungicides like Captan or Myclobutanil. Remove infected leaves.",
        "prevention": "Plant resistant varieties. Ensure good air circulation."
    },
    "Apple___Black_rot": {
        "severity": "High", "cause": "Fungal (Botryosphaeria obtusa)",
        "symptoms": "Brown circular lesions with purple borders on leaves",
        "treatment": "Prune infected branches. Apply copper-based fungicides.",
        "prevention": "Remove mummified fruits. Maintain tree health."
    },
    "Tomato___Late_blight": {
        "severity": "Critical", "cause": "Oomycete (Phytophthora infestans)",
        "symptoms": "Dark brown lesions with white mold on undersides",
        "treatment": "Apply copper fungicides immediately. Remove affected plants.",
        "prevention": "Avoid overhead watering. Use disease-free seeds."
    },
    "Tomato___Early_blight": {
        "severity": "High", "cause": "Fungal (Alternaria solani)",
        "symptoms": "Dark spots with concentric rings (target board pattern)",
        "treatment": "Apply chlorothalonil or mancozeb fungicides.",
        "prevention": "Crop rotation. Remove plant debris after harvest."
    },
    "Potato___Late_blight": {
        "severity": "Critical", "cause": "Oomycete (Phytophthora infestans)",
        "symptoms": "Water-soaked lesions turning dark brown/black",
        "treatment": "Apply metalaxyl or cymoxanil fungicides immediately.",
        "prevention": "Use certified disease-free seed potatoes."
    },
    "Corn_(maize)___Common_rust_": {
        "severity": "Medium", "cause": "Fungal (Puccinia sorghi)",
        "symptoms": "Small oval cinnamon-brown pustules on leaves",
        "treatment": "Apply triazole fungicides. Plant resistant hybrids.",
        "prevention": "Early planting. Monitor fields regularly."
    },
    "Grape___Black_rot": {
        "severity": "High", "cause": "Fungal (Guignardia bidwellii)",
        "symptoms": "Brown circular spots with black dots on leaves",
        "treatment": "Apply mancozeb or myclobutanil fungicides.",
        "prevention": "Remove infected berries. Ensure good canopy airflow."
    },
}

SEVERITY_COLORS = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}

st.set_page_config(
    page_title="🌿 Crop Disease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
        --forest: #183d35;
        --leaf: #45b486;
        --mint: #b9ead2;
        --paper: #0d1218;
        --panel: #151d25;
        --panel-raised: #1b2730;
        --ink: #eef7f2;
        --muted: #a9bab3;
        --line: #2b3d43;
        --amber: #e2a657;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp {
        background: var(--paper);
        background-image: linear-gradient(rgba(69,180,134,.045) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(69,180,134,.045) 1px, transparent 1px);
        background-size: 32px 32px;
    }
    .block-container { max-width: 1440px; padding: 2.5rem 4rem 3rem; }
    .main-header {
        background: linear-gradient(120deg, #173f35 0%, #245f4e 64%, #327d67 100%);
        padding: 2.8rem 3rem 2.5rem; border-radius: 12px; text-align: left;
        margin-bottom: 1.5rem; box-shadow: 0 14px 35px rgba(23,63,53,.16);
        position: relative; overflow: hidden;
    }
    .main-header:after {
        content: ''; position: absolute; right: -5rem; top: -8rem; width: 24rem; height: 24rem;
        border: 1px solid rgba(220,239,229,.22); border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(220,239,229,.05), 0 0 0 56px rgba(220,239,229,.04);
    }
    .main-header h1 { color: white; font-family: 'Space Grotesk', sans-serif; font-size: 2.7rem; letter-spacing: 0; margin: 0; position: relative; z-index: 1; }
    .main-header p { color: #c4e3d2; font-size: 1rem; margin: .7rem 0 0; position: relative; z-index: 1; }
    h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: var(--mint); letter-spacing: 0; }
    h3 { font-size: 1.15rem; }
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stMetric"] { border: 1px solid var(--line); border-radius: 10px; padding: 1rem 1.15rem; box-shadow: 0 8px 22px rgba(0,0,0,.2); transition: transform .2s ease, border-color .2s ease; }
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stMetric"]:hover { transform: translateY(-3px); border-color: var(--leaf); }
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stColumn"]:nth-child(1) [data-testid="stMetric"] { background: linear-gradient(145deg, #173b37, #1c5146); }
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stColumn"]:nth-child(2) [data-testid="stMetric"] { background: linear-gradient(145deg, #192f49, #214568); }
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] { background: linear-gradient(145deg, #49321e, #684522); }
    [data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] { background: linear-gradient(145deg, #392743, #563765); }
    [data-testid="stMetricLabel"] { color: #d3e6dc; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
    [data-testid="stMetricValue"] { color: #ffffff; font-family: 'Space Grotesk', sans-serif; }
    [data-testid="stMetricDelta"] { color: var(--mint) !important; }
    .stButton > button, .stDownloadButton > button { border-radius: 8px; border: 1px solid #367d67; background: var(--forest); color: white; font-weight: 600; padding: .55rem 1rem; transition: background .2s ease, transform .2s ease; }
    .stButton > button:hover, .stDownloadButton > button:hover { background: var(--leaf); border-color: var(--leaf); color: white; transform: translateY(-1px); }
    [data-testid="stFileUploader"] { background: var(--panel); border: 1px dashed #5ebc98; border-radius: 10px; padding: .65rem; }
    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] label p { color: #f3fff8 !important; font-weight: 600 !important; }
    [data-testid="stFileUploader"] section { background: var(--panel-raised); border-radius: 8px; }
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: #b8cbc2 !important; }
    [data-testid="stTabs"] { margin-top: 1.5rem; }
    [data-testid="stTabs"] button { color: var(--muted); font-weight: 600; padding: .75rem 1rem; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--mint); }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: var(--leaf); }
    [data-testid="stProgressBar"] > div > div { background: var(--leaf); }
    [data-testid="stAlert"] { border-radius: 9px; background: var(--panel-raised); color: var(--ink); border: 1px solid var(--line); }
    .result-card {
        border-radius: 10px; padding: 1.7rem 2rem; color: white;
        text-align: left; box-shadow: 0 10px 24px rgba(23,63,53,.14); margin: 1rem 0;
    }
    .disease-card {
        background: var(--panel); border-radius: 10px; padding: 1.35rem;
        border: 1px solid var(--line); margin: 0.5rem 0; box-shadow: 0 5px 15px rgba(23,63,53,.04);
    }
    .batch-card {
        background: var(--panel); border-radius: 10px; padding: 1rem;
        border-left: 3px solid var(--leaf); margin: 0.5rem 0;
        box-shadow: 0 5px 15px rgba(23,63,53,.07);
    }
    div[data-testid="stSidebar"] { background: #111b20; border-right: 1px solid rgba(255,255,255,.08); }
    div[data-testid="stSidebar"] * { color: white !important; }
    div[data-testid="stSidebar"] hr { border-color: rgba(220,239,229,.2); }
    div[data-testid="stSidebar"] [data-baseweb="radio"] label, div[data-testid="stSidebar"] [data-baseweb="checkbox"] label { color: #d6e8de !important; }
    @media (max-width: 800px) {
        .block-container { padding: 1.2rem 1rem 2rem; }
        .main-header { padding: 2rem 1.4rem; }
        .main-header h1 { font-size: 2rem; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    with open(NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names

def get_gradcam(model, img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer("out_relu").output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap):
    heatmap_resized = np.array(Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE))
    heatmap_colored = mpl_cm.jet(heatmap_resized / 255.0)[:, :, :3]
    original = np.array(img) / 255.0
    return heatmap_colored * 0.4 + original * 0.6

def predict_image(model, class_names, img):
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0).astype(np.float32)
    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions[0])
    pred_class = class_names[pred_idx]
    confidence = float(predictions[0][pred_idx]) * 100
    top5 = [(class_names[i], float(predictions[0][i]) * 100)
            for i in np.argsort(predictions[0])[::-1][:5]]
    margin = (top5[0][1] - top5[1][1]) if len(top5) > 1 else confidence
    return pred_class, confidence, top5, margin, img_array

def assess_image_quality(img):
    pixels = np.asarray(img.convert("L"), dtype=np.float32)
    brightness = float(pixels.mean())
    contrast = float(pixels.std())
    gy, gx = np.gradient(pixels)
    sharpness = float((gx * gx + gy * gy).mean())
    issues = []
    if brightness < 45:
        issues.append("too dark")
    elif brightness > 220:
        issues.append("overexposed")
    if contrast < 18:
        issues.append("low contrast")
    if sharpness < 120:
        issues.append("possibly blurry")
    score = max(0, min(100, 100 - len(issues) * 25))
    return score, issues

def validate_leaf_image(img):
    """Reject obvious non-plant photos before the closed-set classifier runs."""
    pixels = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    green_pixels = (green > red * 1.05) & (green > blue * 1.02) & (green > 0.18)
    foliage_ratio = float(green_pixels.mean())
    height, width = green_pixels.shape
    center = green_pixels[int(height * 0.15):int(height * 0.85), int(width * 0.15):int(width * 0.85)]
    center_foliage_ratio = float(center.mean())
    if foliage_ratio < 0.06 or center_foliage_ratio < 0.08:
        return False, "This does not look like a leaf image. Upload a clear photo of one plant leaf."
    return True, "Leaf image detected"

def get_ai_guidance(pred_class, confidence, margin, quality_score, quality_issues, threshold):
    if quality_issues:
        return ("Retake recommended", "Improve the photo: " + ", ".join(quality_issues) + ". "
                "Use daylight, hold the camera steady, and fill the frame with one leaf.")
    if confidence < threshold or margin < 10:
        return ("Needs human review", "The model is uncertain between similar classes. "
                "Capture another clear image and confirm the result with a local agricultural expert.")
    if "healthy" in pred_class.lower():
        return ("Monitor", "No visible disease pattern was detected. Continue regular scouting and recheck new symptoms.")
    return ("Act and monitor", "Isolate visibly affected leaves, follow the disease guidance below, and verify the diagnosis before applying chemicals.")

def generate_pdf_report(results, summary):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(27, 67, 50)
    pdf.rect(0, 0, 210, 40, 'F')
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, "", ln=True)
    pdf.cell(0, 10, "Crop Disease Detection Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(149, 213, 178)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

    # Summary
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_fill_color(240, 255, 240)
    pdf.cell(0, 10, "Summary", ln=True, fill=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.ln(3)
    pdf.cell(0, 8, f"Total Images Analyzed: {summary['total']}", ln=True)
    pdf.cell(0, 8, f"Healthy Plants: {summary['healthy']}", ln=True)
    pdf.cell(0, 8, f"Diseased Plants: {summary['diseased']}", ln=True)
    pdf.cell(0, 8, f"Average Confidence: {summary['avg_conf']:.1f}%", ln=True)

    # Results
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_fill_color(240, 255, 240)
    pdf.cell(0, 10, "Detailed Results", ln=True, fill=True)
    pdf.ln(3)

    for i, r in enumerate(results):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_fill_color(27, 67, 50)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 8, f"Image {i+1}: {r['filename']}", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 7, f"  Prediction: {r['disease']} on {r['crop']}", ln=True)
        pdf.cell(0, 7, f"  Confidence: {r['confidence']:.1f}%", ln=True)
        pdf.cell(0, 7, f"  Status: {'Healthy' if r['is_healthy'] else 'Diseased'}", ln=True)

        db = DISEASE_DB.get(r['pred_class'])
        if db:
            pdf.cell(0, 7, f"  Severity: {db['severity']}", ln=True)
            pdf.cell(0, 7, f"  Treatment: {db['treatment']}", ln=True)
        pdf.ln(4)

    # Footer
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Built by Rekha Munusamy | MobileNetV2 + Grad-CAM | PlantVillage Dataset | 92.55% Accuracy", ln=True, align="C")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

def get_download_link(file_path, label):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="crop_disease_report.pdf">📄 {label}</a>'

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Crop Disease AI")
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("**Model:** MobileNetV2")
    st.markdown("**Accuracy:** 92.55%")
    st.markdown("**Classes:** 38 diseases")
    st.markdown("**Crops:** 14 types")
    st.markdown("---")
    st.markdown("### 🌾 Supported Crops")
    for crop in ["🍎 Apple","🫐 Blueberry","🍒 Cherry","🌽 Corn","🍇 Grape",
                 "🍊 Orange","🍑 Peach","🫑 Pepper","🥔 Potato","🍓 Strawberry",
                 "🍅 Tomato","🌿 Soybean","🎃 Squash","🍃 Raspberry"]:
        st.markdown(f"  {crop}")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    mode = st.radio("Mode", ["Single Image", "Batch Upload"])
    show_gradcam = st.checkbox("Show Grad-CAM", value=True)
    show_top5 = st.checkbox("Show Top 5 Predictions", value=True)
    confidence_threshold = st.slider("Confidence Threshold %", 0, 100, 70)
    st.markdown("---")
    st.markdown("**Built by Rekha Munusamy**")
    st.markdown("*B.E. CSE (AIML) — SKCT*")

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>Field Lens <span style='color:#a9d9bd;'>/</span> Crop health intelligence</h1>
    <p>Upload a leaf, inspect the prediction, then see exactly where the model looked.</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("🎯 Accuracy", "92.55%", "+4.2%")
with m2: st.metric("🌿 Classes", "38", "14 crops")
with m3: st.metric("🖼️ Training Images", "54,305", "PlantVillage")
with m4: st.metric("🧠 Architecture", "MobileNetV2", "Transfer Learning")
st.markdown("---")

model, class_names = load_resources()
uploaded_file = None
uploaded_files = []
single_prediction = None
batch_results = []
upload_is_valid = False
upload_validation_message = ""

upload_tab, prediction_tab, gradcam_tab, about_tab = st.tabs([
    "📤 Upload", "🔎 Prediction", "🗺️ Grad-CAM Visualization", "ℹ️ About"
])

with upload_tab:
    if mode == "Single Image":
        st.markdown("### Add a leaf image")
        uploaded_file = st.file_uploader(
            "Choose a clear JPG or PNG leaf image",
            type=["jpg", "jpeg", "png"],
            key="single_upload"
        )
        if uploaded_file:
            upload_img = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
            upload_is_valid, upload_validation_message = validate_leaf_image(upload_img)
            left, right = st.columns([1, 1.5])
            with left:
                st.image(upload_img, caption=uploaded_file.name, width="stretch")
            with right:
                if upload_is_valid:
                    st.success(upload_validation_message)
                    st.markdown("Use the **Prediction** tab for the diagnosis and the **Grad-CAM Visualization** tab to inspect model attention.")
                else:
                    st.error(upload_validation_message)
                    st.caption("The classifier only understands PlantVillage-style leaf photos and cannot safely label unrelated images.")
        else:
            st.info("Start by uploading one leaf image. For best results, use daylight and fill the frame with a single leaf.")
    else:
        st.markdown("### Analyze a group of leaves")
        uploaded_files = st.file_uploader(
            "Choose up to 10 JPG or PNG leaf images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_upload"
        )
        if len(uploaded_files) > 10:
            st.warning("Only the first 10 images will be analyzed.")
            uploaded_files = uploaded_files[:10]
        if uploaded_files:
            st.success(f"{len(uploaded_files)} image(s) ready for analysis")
            preview_cols = st.columns(min(len(uploaded_files), 5))
            for index, file in enumerate(uploaded_files[:5]):
                with preview_cols[index]:
                    st.image(Image.open(file).convert("RGB"), caption=file.name, width="stretch")
            st.caption("Open the Prediction tab to process the batch and download reports.")
        else:
            st.info("Select multiple leaf images to create a field-level summary with CSV and PDF reports.")

with prediction_tab:
    if mode == "Single Image":
        if uploaded_file and upload_is_valid:
            img = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
            pred_class, confidence, top5, margin, img_array = predict_image(model, class_names, img)
            single_prediction = (pred_class, confidence, top5, margin, img, img_array)
            crop = pred_class.split("___")[0].replace("_", " ")
            disease = pred_class.split("___")[1].replace("_", " ") if "___" in pred_class else pred_class
            is_healthy = "healthy" in pred_class.lower()
            quality_score, quality_issues = assess_image_quality(img)
            guidance_title, guidance = get_ai_guidance(pred_class, confidence, margin, quality_score, quality_issues, confidence_threshold)
            status_color = "#1a7a4a" if is_healthy else "#a93226"
            status_title = "HEALTHY PLANT" if is_healthy else "DISEASE DETECTED"
            st.markdown(f"""<div class='result-card' style='background:{status_color};'>
                <h2>{'✅' if is_healthy else '⚠️'} {status_title}</h2>
                <h3>{crop}{'' if is_healthy else ' — ' + disease}</h3>
                <h4>Confidence: {confidence:.1f}%</h4>
            </div>""", unsafe_allow_html=True)
            st.markdown("### Confidence and decision support")
            st.progress(int(confidence))
            if confidence >= confidence_threshold:
                st.success(f"High confidence ({confidence:.1f}%)")
            else:
                st.warning(f"Low confidence ({confidence:.1f}%). Retake the photo if possible.")
            ai1, ai2, ai3 = st.columns(3)
            with ai1: st.metric("Photo quality", f"{quality_score:.0f}/100")
            with ai2: st.metric("Top-2 margin", f"{margin:.1f}%")
            with ai3: st.metric("Next action", guidance_title)
            (st.warning if quality_issues or confidence < confidence_threshold or margin < 10 else st.info)(guidance)

            db = DISEASE_DB.get(pred_class)
            if db:
                st.markdown(f"### {SEVERITY_COLORS.get(db['severity'], '🟡')} Severity: **{db['severity']}**")
                info_left, info_right = st.columns(2)
                with info_left:
                    st.markdown(f"**Cause**\n\n{db['cause']}\n\n**Symptoms**\n\n{db['symptoms']}")
                with info_right:
                    st.markdown(f"**Treatment**\n\n{db['treatment']}\n\n**Prevention**\n\n{db['prevention']}")
            if show_top5:
                st.markdown("### Top predictions")
                for index, (name, probability) in enumerate(top5):
                    label = name.replace("___", " → ").replace("_", " ")
                    st.progress(int(probability), text=f"#{index + 1}  {label}: {probability:.2f}%")
        elif uploaded_file:
            st.error(f"Invalid input: {upload_validation_message}")
            st.info("Return to the Upload tab and choose a close-up photo of a plant leaf.")
        else:
            st.info("Upload an image in the Upload tab to see the prediction here.")
    elif uploaded_files:
        st.markdown("### Batch prediction results")
        progress_bar = st.progress(0)
        for index, file in enumerate(uploaded_files):
            img = Image.open(file).convert("RGB").resize(IMG_SIZE)
            pred_class, confidence, top5, margin, _ = predict_image(model, class_names, img)
            crop = pred_class.split("___")[0].replace("_", " ")
            disease = pred_class.split("___")[1].replace("_", " ") if "___" in pred_class else pred_class
            is_healthy = "healthy" in pred_class.lower()
            quality_score, quality_issues = assess_image_quality(img)
            guidance_title, _ = get_ai_guidance(pred_class, confidence, margin, quality_score, quality_issues, confidence_threshold)
            batch_results.append({"filename": file.name, "pred_class": pred_class, "crop": crop, "disease": disease, "confidence": confidence, "margin": margin, "quality": quality_score, "action": guidance_title, "is_healthy": is_healthy, "timestamp": datetime.now().strftime("%H:%M:%S")})
            col_img, col_res = st.columns([1, 2])
            with col_img: st.image(img, caption=file.name, width="stretch")
            with col_res:
                (st.success if is_healthy else st.error)(f"{'✅ HEALTHY' if is_healthy else '⚠️ ' + disease} — {crop} ({confidence:.1f}%)")
                st.progress(int(confidence))
                st.caption(f"AI action: {guidance_title} | Photo quality: {quality_score:.0f}/100")
            progress_bar.progress((index + 1) / len(uploaded_files))
        df = pd.DataFrame(batch_results)[["filename", "crop", "disease", "confidence", "margin", "quality", "action", "is_healthy", "timestamp"]]
        df.columns = ["File", "Crop", "Disease", "Confidence %", "Top-2 Margin %", "Photo Quality", "AI Action", "Healthy", "Time"]
        df["Confidence %"] = df["Confidence %"].round(2)
        df["Top-2 Margin %"] = df["Top-2 Margin %"].round(2)
        df["Photo Quality"] = df["Photo Quality"].round(0).astype(int)
        st.dataframe(df, width="stretch")
        total = len(batch_results)
        healthy = sum(result["is_healthy"] for result in batch_results)
        s1, s2, s3, s4 = st.columns(4)
        with s1: st.metric("Total", total)
        with s2: st.metric("Healthy", healthy)
        with s3: st.metric("Diseased", total - healthy)
        with s4: st.metric("Avg confidence", f"{df['Confidence %'].mean():.1f}%")
        st.download_button("📥 Download CSV Report", df.to_csv(index=False), file_name=f"crop_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
        if st.button("📄 Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                summary = {"total": total, "healthy": healthy, "diseased": total - healthy, "avg_conf": df["Confidence %"].mean()}
                pdf_path = generate_pdf_report(batch_results, summary)
                st.markdown(get_download_link(pdf_path, "Download PDF Report"), unsafe_allow_html=True)
                st.success("PDF generated")
    else:
        st.info("Upload images in the Upload tab to see batch results here.")

with gradcam_tab:
    if mode == "Single Image" and single_prediction:
        pred_class, confidence, top5, margin, img, img_array = single_prediction
        if show_gradcam:
            with st.spinner("Generating Grad-CAM visualization..."):
                heatmap = get_gradcam(model, img_array)
                overlay = overlay_gradcam(img, heatmap)
            c1, c2, c3 = st.columns(3)
            with c1: st.image(img, caption="Original image", width="stretch")
            with c2:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(np.array(Image.fromarray(np.uint8(255 * heatmap)).resize(IMG_SIZE)), cmap="jet")
                ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)
            with c3: st.image(overlay, caption="Model attention overlay", width="stretch")
            st.info("Red areas show where the model focused most; blue areas contributed less to the prediction.")
        else:
            st.info("Enable Show Grad-CAM in the sidebar to generate the visualization.")
    else:
        st.info("Grad-CAM is available for single-image analysis. Upload one image and open Prediction first.")

with about_tab:
    st.markdown("### About Field Lens")
    st.markdown("This crop health tool uses transfer learning to identify plant disease patterns from leaf images and Grad-CAM to make its visual reasoning easier to inspect.")
    about_left, about_right = st.columns(2)
    with about_left:
        st.markdown("#### Model details")
        st.markdown("**Architecture:** MobileNetV2  \n**Dataset:** PlantVillage  \n**Classes:** 38 disease and healthy-leaf categories  \n**Reported accuracy:** 92.55%")
    with about_right:
        st.markdown("#### Use responsibly")
        st.markdown("Treat predictions as decision support. For low-confidence results or high-impact treatment decisions, capture another image and consult a local agricultural expert.")

st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:1rem;background:#151d25;border:1px solid #2b3d43;border-radius:10px;'>
    <p style='color:#b9ead2;font-weight:bold;'>🌿 Crop Disease Detection System</p>
    <p style='color:#a9bab3;font-size:0.9rem;'>Built by Rekha Munusamy | B.E. CSE (AIML) | Sri Krishna College of Technology</p>
    <p style='color:#81958d;font-size:0.8rem;'>MobileNetV2 + Transfer Learning + Grad-CAM | PlantVillage Dataset | 92.55% Accuracy</p>
</div>""", unsafe_allow_html=True)