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
    .main-header {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 50%, #40916C 100%);
        padding: 2rem; border-radius: 15px; text-align: center;
        margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: white; font-size: 2.5rem; margin: 0; }
    .main-header p { color: #95D5B2; font-size: 1.1rem; margin: 0.5rem 0 0; }
    .result-card {
        border-radius: 15px; padding: 1.5rem; color: white;
        text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin: 1rem 0;
    }
    .disease-card {
        background: #f8fff8; border-radius: 12px; padding: 1.5rem;
        border: 1px solid #95D5B2; margin: 0.5rem 0;
    }
    .batch-card {
        background: white; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #2D6A4F; margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    div[data-testid="stSidebar"] { background: #1B4332; }
    div[data-testid="stSidebar"] * { color: white !important; }
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
    return pred_class, confidence, top5, img_array

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
    <h1>🌿 Crop Disease Detection System</h1>
    <p>AI-powered plant disease detection using MobileNetV2 + Grad-CAM Explainability</p>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("🎯 Accuracy", "92.55%", "+4.2%")
with m2: st.metric("🌿 Classes", "38", "14 crops")
with m3: st.metric("🖼️ Training Images", "54,305", "PlantVillage")
with m4: st.metric("🧠 Architecture", "MobileNetV2", "Transfer Learning")
st.markdown("---")

model, class_names = load_resources()

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE IMAGE MODE
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Single Image":
    col_upload, col_result = st.columns([1, 2])

    with col_upload:
        st.markdown("### 📤 Upload Leaf Image")
        uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg","jpeg","png"])
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB").resize(IMG_SIZE)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            st.success(f"✅ {uploaded_file.name}")

    with col_result:
        if uploaded_file:
            pred_class, confidence, top5, img_array = predict_image(model, class_names, img)
            crop = pred_class.split("___")[0].replace("_"," ")
            disease = pred_class.split("___")[1].replace("_"," ") if "___" in pred_class else pred_class
            is_healthy = "healthy" in pred_class.lower()

            if is_healthy:
                st.markdown(f"""
                <div class='result-card' style='background:linear-gradient(135deg,#1a7a4a,#2ecc71);'>
                    <h2>✅ HEALTHY PLANT</h2><h3>{crop}</h3>
                    <h4>Confidence: {confidence:.1f}%</h4>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-card' style='background:linear-gradient(135deg,#c0392b,#922b21);'>
                    <h2>⚠️ DISEASE DETECTED</h2><h3>{crop} — {disease}</h3>
                    <h4>Confidence: {confidence:.1f}%</h4>
                </div>""", unsafe_allow_html=True)

            st.markdown("### 📊 Confidence Level")
            st.progress(int(confidence))
            if confidence >= confidence_threshold:
                st.success(f"✅ High confidence ({confidence:.1f}%)")
            else:
                st.warning(f"⚠️ Low confidence ({confidence:.1f}%) — retake the photo")

            st.markdown("---")
            tab1, tab2, tab3 = st.tabs(["🗺️ Grad-CAM", "💊 Disease Info", "📈 Top Predictions"])

            with tab1:
                if show_gradcam:
                    with st.spinner("Generating Grad-CAM..."):
                        heatmap = get_gradcam(model, img_array)
                        overlay = overlay_gradcam(img, heatmap)
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**Original**")
                        st.image(img, use_container_width=True)
                    with c2:
                        st.markdown("**Heatmap**")
                        fig, ax = plt.subplots(figsize=(4,4))
                        ax.imshow(np.array(Image.fromarray(np.uint8(255*heatmap)).resize(IMG_SIZE)), cmap="jet")
                        ax.axis("off")
                        st.pyplot(fig)
                    with c3:
                        st.markdown("**Overlay**")
                        st.image(overlay, use_container_width=True)
                    st.info("🔴 Red = AI focused here | 🔵 Blue = less important")

            with tab2:
                db = DISEASE_DB.get(pred_class)
                if db:
                    icon = SEVERITY_COLORS.get(db["severity"], "🟡")
                    st.markdown(f"### {icon} Severity: **{db['severity']}**")
                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown(f"""<div class='disease-card'>
                            <h4>🦠 Cause</h4><p>{db['cause']}</p>
                            <h4>👁️ Symptoms</h4><p>{db['symptoms']}</p>
                        </div>""", unsafe_allow_html=True)
                    with cb:
                        st.markdown(f"""<div class='disease-card'>
                            <h4>💊 Treatment</h4><p>{db['treatment']}</p>
                            <h4>🛡️ Prevention</h4><p>{db['prevention']}</p>
                        </div>""", unsafe_allow_html=True)
                elif is_healthy:
                    st.success("✅ Plant is healthy! No treatment needed.")
                    st.balloons()
                else:
                    st.warning("**Treatment:** Consult an agricultural expert.")
                    st.info("**Prevention:** Regular monitoring and crop rotation.")

            with tab3:
                if show_top5:
                    for i, (name, prob) in enumerate(top5):
                        n = name.replace("___"," → ").replace("_"," ")
                        cr, cb2 = st.columns([1,4])
                        with cr: st.markdown(f"**#{i+1}**")
                        with cb2: st.progress(int(prob), text=f"{n}: {prob:.2f}%")

        else:
            st.markdown("""
            <div style='text-align:center;padding:80px;background:#f0fff4;border-radius:15px;border:2px dashed #2D6A4F;'>
                <h2>👈 Upload a leaf image</h2>
                <p>Get instant disease detection with AI explainability</p>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BATCH UPLOAD MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("### 📦 Batch Upload — Analyze Multiple Leaves at Once")
    st.info("Upload up to 10 leaf images at once. Results will be shown for each image with a downloadable CSV and PDF report.")

    uploaded_files = st.file_uploader(
        "Upload multiple leaf images (max 10)",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) > 10:
            st.error("❌ Maximum 10 images allowed at once!")
            uploaded_files = uploaded_files[:10]

        st.markdown(f"**{len(uploaded_files)} image(s) uploaded — Processing...**")
        progress_bar = st.progress(0)

        results = []
        batch_tab1, batch_tab2 = st.tabs(["📊 Results Grid", "📋 Summary Table"])

        with batch_tab1:
            for i, file in enumerate(uploaded_files):
                img = Image.open(file).convert("RGB").resize(IMG_SIZE)
                pred_class, confidence, top5, img_array = predict_image(model, class_names, img)
                crop = pred_class.split("___")[0].replace("_"," ")
                disease = pred_class.split("___")[1].replace("_"," ") if "___" in pred_class else pred_class
                is_healthy = "healthy" in pred_class.lower()

                results.append({
                    "filename": file.name,
                    "pred_class": pred_class,
                    "crop": crop,
                    "disease": disease,
                    "confidence": confidence,
                    "is_healthy": is_healthy,
                    "timestamp": datetime.now().strftime("%H:%M:%S")
                })

                col_img, col_res = st.columns([1, 2])
                with col_img:
                    st.image(img, caption=file.name, use_container_width=True)
                with col_res:
                    if is_healthy:
                        st.success(f"✅ **HEALTHY** — {crop} ({confidence:.1f}%)")
                    else:
                        st.error(f"⚠️ **{disease}** on {crop} ({confidence:.1f}%)")
                    st.progress(int(confidence))
                    db = DISEASE_DB.get(pred_class)
                    if db:
                        st.markdown(f"**Severity:** {SEVERITY_COLORS.get(db['severity'],'')} {db['severity']}")
                        st.markdown(f"**Treatment:** {db['treatment']}")

                st.markdown("---")
                progress_bar.progress((i+1)/len(uploaded_files))

        with batch_tab2:
            df = pd.DataFrame(results)[["filename","crop","disease","confidence","is_healthy","timestamp"]]
            df.columns = ["File","Crop","Disease","Confidence %","Healthy","Time"]
            df["Confidence %"] = df["Confidence %"].round(2)
            st.dataframe(df, use_container_width=True)

            # Summary stats
            total = len(results)
            healthy = sum(1 for r in results if r["is_healthy"])
            diseased = total - healthy
            avg_conf = sum(r["confidence"] for r in results) / total

            s1, s2, s3, s4 = st.columns(4)
            with s1: st.metric("Total", total)
            with s2: st.metric("✅ Healthy", healthy)
            with s3: st.metric("⚠️ Diseased", diseased)
            with s4: st.metric("Avg Confidence", f"{avg_conf:.1f}%")

            # CSV Download
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV Report",
                data=csv,
                file_name=f"crop_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

            # PDF Download
            if st.button("📄 Generate PDF Report"):
                with st.spinner("Generating PDF..."):
                    summary = {"total": total, "healthy": healthy, "diseased": diseased, "avg_conf": avg_conf}
                    pdf_path = generate_pdf_report(results, summary)
                    st.markdown(get_download_link(pdf_path, "Download PDF Report"), unsafe_allow_html=True)
                    st.success("✅ PDF generated!")

    else:
        st.markdown("""
        <div style='text-align:center;padding:60px;background:#f0fff4;border-radius:15px;border:2px dashed #2D6A4F;'>
            <h2>📤 Upload multiple leaf images</h2>
            <p>Select up to 10 images at once for batch analysis</p>
            <p>Get CSV + PDF report with all predictions!</p>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:1rem;background:#f0fff4;border-radius:10px;'>
    <p style='color:#2D6A4F;font-weight:bold;'>🌿 Crop Disease Detection System</p>
    <p style='color:#555;font-size:0.9rem;'>Built by Rekha Munusamy | B.E. CSE (AIML) | Sri Krishna College of Technology</p>
    <p style='color:#555;font-size:0.8rem;'>MobileNetV2 + Transfer Learning + Grad-CAM | PlantVillage Dataset | 92.55% Accuracy</p>
</div>""", unsafe_allow_html=True)