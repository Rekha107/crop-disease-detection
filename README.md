# 🌿 Crop Disease Detection System
### AI-powered plant disease detection using MobileNetV2 + Grad-CAM Explainability

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=for-the-badge&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-92.55%25-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📌 Project Overview

A deep learning-based web application that detects crop diseases from leaf images using **MobileNetV2 Transfer Learning** and provides explainable AI results through **Grad-CAM heatmaps**. Built to help farmers identify plant diseases early and take timely action to prevent crop loss.

> 🌾 **Real-world impact:** Farmers in India lose 20–30% of crop yield annually due to undetected plant diseases. This system provides instant AI-powered diagnosis from a simple leaf photo.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Single Image Detection** | Upload a leaf image → get instant disease prediction with confidence score |
| 🗺️ **Grad-CAM Visualization** | Heatmap showing exactly which part of the leaf the AI focused on |
| 📦 **Batch Processing** | Upload up to 10 images at once for bulk analysis |
| 📥 **CSV Export** | Download all predictions as a structured CSV report |
| 📄 **PDF Report** | Generate professional PDF report with predictions, severity, and treatment |
| 📊 **Top-5 Predictions** | See confidence scores for top 5 possible diseases |
| ⚙️ **Confidence Threshold** | Adjustable confidence filter to control prediction sensitivity |

---

## 🧠 Model Architecture

```
Input Image (224×224×3)
        ↓
MobileNetV2 (pretrained on ImageNet — frozen base)
        ↓
Global Average Pooling 2D
        ↓
Dense (256, ReLU) + BatchNorm + Dropout(0.5)
        ↓
Dense (38, Softmax) → Disease Classification
```

### Why MobileNetV2?
- **Lightweight** — only 3.4M parameters vs VGG16's 138M
- **Pretrained** — trained on 1.2M ImageNet images, reuses visual feature knowledge
- **Mobile-ready** — designed for real-world deployment on mobile devices
- **High accuracy** — achieves 92.55% on PlantVillage with fine-tuning

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | **92.55%** |
| Macro Avg Precision | 92% |
| Macro Avg Recall | 90% |
| Macro Avg F1-Score | 91% |
| Training Images | 43,444 |
| Validation Images | 10,861 |
| Total Classes | 38 |

---

## 🌾 Dataset

**PlantVillage Dataset** (Kaggle)
- **Total images:** 54,305 (color subset)
- **Classes:** 38 disease categories
- **Crops covered:** 14 crop types
- **Source:** [Kaggle — abdallahalidev/plantvillage-dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

### Crops Supported
🍎 Apple | 🫐 Blueberry | 🍒 Cherry | 🌽 Corn | 🍇 Grape | 🍊 Orange | 🍑 Peach | 🫑 Pepper | 🥔 Potato | 🍓 Strawberry | 🍅 Tomato | 🌿 Soybean | 🎃 Squash | 🍃 Raspberry

---

## 🗺️ Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which regions of the leaf the model focused on to make its prediction.

- 🔴 **Red areas** = model focused heavily here (disease spots)
- 🟡 **Yellow areas** = moderate attention
- 🔵 **Blue areas** = less important regions

This makes the model **explainable and trustworthy** — farmers can verify the AI is looking at the right areas.

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.11
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Rekha107/crop-disease-detection.git
cd crop-disease-detection

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
crop-disease-detection/
│
├── app.py                  # Streamlit web application
├── train.py                # Model training script
├── evaluate.py             # Model evaluation + confusion matrix
├── gradcam.py              # Grad-CAM visualization script
├── best_model.h5           # Trained MobileNetV2 model
├── class_names.json        # 38 disease class names
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🔬 Project Phases

```
Phase 1 → Dataset EDA (PlantVillage, 54K images, 38 classes)
    ↓
Phase 2 → Preprocessing + Augmentation (flip, rotate, zoom, normalize)
    ↓
Phase 3 → MobileNetV2 Transfer Learning (frozen base → fine-tune)
    ↓
Phase 4 → Evaluation (confusion matrix, classification report, Grad-CAM)
    ↓
Phase 5 → Streamlit Web App (single + batch + CSV + PDF)
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.11 | Core language |
| TensorFlow 2.13 | Deep learning framework |
| Keras | Model building API |
| MobileNetV2 | Transfer learning backbone |
| Streamlit | Web application framework |
| Matplotlib | Visualization |
| Grad-CAM | Explainability |
| scikit-learn | Metrics & evaluation |
| fpdf2 | PDF report generation |
| pandas | Data handling |

---

## 📈 Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | MobileNetV2 (ImageNet weights) |
| Input Size | 224 × 224 × 3 |
| Batch Size | 32 |
| Optimizer | Adam (lr=0.001) |
| Loss | Categorical Crossentropy |
| Epochs | 10 |
| Early Stopping | patience=5 |
| Data Split | 80% train / 20% val |
| Augmentation | flip, rotate, zoom, shift |

---

## 👩‍💻 Author

**Rekha Munusamy**
- 🎓 B.E. CSE (AIML) — Sri Krishna College of Technology, Coimbatore
- 🆔 Student ID: 727824TUAM037
- 💼 [LinkedIn](https://linkedin.com/in/rekha-munusamy)
- 🐙 [GitHub](https://github.com/Rekha107)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) by Abdallah Ali
- [MobileNetV2](https://arxiv.org/abs/1801.04381) by Google
- [Grad-CAM](https://arxiv.org/abs/1610.02391) by Selvaraju et al.

---

<p align="center">
  <b>🌿 Built with ❤️ to help farmers detect crop diseases early and save harvests</b>
</p>
