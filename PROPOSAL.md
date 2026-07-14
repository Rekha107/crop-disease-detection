# Project Proposal — Crop Disease Detection System

**Student:** Rekha Munusamy
**Roll No:** 727824TUAM037
**Course:** 23ADC04 Deep Learning
**Repository:** https://github.com/Rekha107/crop-disease-detection

---

## 1. Problem Statement

Crop diseases cause an estimated 20–30% annual yield loss for Indian farmers, largely because diseases are detected too late using manual visual inspection, which requires expert knowledge that is often not accessible in rural areas. There is a need for a fast, low-cost, accessible tool that can diagnose plant diseases directly from a leaf photograph.

## 2. Objective

To build a deep learning-based image classification system that:
- Detects and classifies plant leaf diseases across 38 categories spanning 14 crop types, using a single leaf image as input.
- Provides visual explainability (via Grad-CAM) so the prediction is trustworthy and interpretable, not a black box.
- Is deployable as an accessible web application usable by non-technical end users (farmers, agri-extension workers).

**Expected outcome:** A trained classifier achieving >90% validation accuracy, wrapped in a Streamlit web app supporting single-image and batch prediction, with downloadable CSV/PDF reports.

## 3. Dataset Source

- **Name:** PlantVillage Dataset
- **Source:** Kaggle — `abdallahalidev/plantvillage-dataset`
- **Size:** 54,305 images (color subset), split into 43,444 training / 10,861 validation images
- **Classes:** 38 disease/healthy categories across 14 crop species (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Strawberry, Tomato, Soybean, Squash, Raspberry)

## 4. Proposed Architecture / Methodology

- **Backbone:** MobileNetV2, pretrained on ImageNet, used as a frozen feature extractor (transfer learning).
- **Head:** Global Average Pooling → Dense(256, ReLU) + BatchNorm + Dropout(0.5) → Dense(38, Softmax)
- **Why MobileNetV2:** Lightweight (3.4M params vs. VGG16's 138M), mobile/edge-deployment friendly, strong accuracy on PlantVillage-style datasets in published literature.
- **Explainability:** Grad-CAM heatmaps to visualize which leaf regions drove each prediction.
- **Deployment:** Streamlit web application with single-image detection, batch upload (up to 10 images), CSV export, and automated PDF report generation.

## 5. Real-World Relevance

This directly addresses a documented gap in Indian agriculture: lack of accessible, instant, explainable disease diagnosis tools for farmers who cannot access expert pathologists in time to prevent crop loss.
