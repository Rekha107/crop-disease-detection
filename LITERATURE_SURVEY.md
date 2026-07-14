# Literature Survey — Crop Disease Detection using Deep Learning

**Student:** Rekha Munusamy | **Roll No:** 727824TUAM037

## Survey Table

| # | Paper Title | Year | Method | Result | Gap Identified |
|---|---|---|---|---|---|
| 1 | Plant Leaf Disease Detection using Transfer Learning and Explainable AI [1] | 2022 | Compared EfficientNetV2L, MobileNetV2, ResNet152V2 with transfer learning | High classification accuracy across multiple pretrained backbones | Focuses on accuracy comparison only; explainability is mentioned but not deeply integrated into a deployable end-user tool |
| 2 | Plant Disease Detection using CNN and Transfer Learning [2] | 2021 | VGG16, Xception, InceptionV3, ResNet152, MobileNetV2 on PlantVillage + PlantDoc | Multi-class classification across two datasets, disease severity index proposed | No lightweight deployment strategy; heavier models (VGG16, ResNet152) are impractical for edge/mobile use |
| 3 | Deep Learning-Based Transfer Learning with MobileNetV2 for Crop Disease Detection [3] | 2025 | MobileNetV2 as frozen feature extractor with global average pooling, on Kaggle apple disease dataset | Effective classification restricted to apple crop diseases | Narrow crop scope (single crop); no batch processing or reporting pipeline for practical use |
| 4 | Transfer Learning Approach for Plant Leaf Disease Detection Using CNN with MobileNetV2 [4] | 2020 | MobileNetV2 pretrained feature extraction, CNN classifier, focused on 5 major Bangladesh crops | Demonstrated feasibility for low-resource farming context | No explainability (Grad-CAM/heatmaps); model output is a bare label with no visual justification for the farmer |
| 5 | Plant Leaf Disease Detection using MobileNetV2 [5] | 2025 | MobileNetV2 transfer learning on PlantVillage | Strong accuracy on PlantVillage benchmark | Paper is model/accuracy-focused; lacks discussion of deployment, batch inference, or exportable reports for end users |
| 6 | LeafDoc-Net: Robust Lightweight Transfer Learning Architecture for Leaf Disease Detection [6] | 2024 | Concatenated DenseNet121 + MobileNetV2 with attention modules, evaluated with Grad-CAM++ | Superior accuracy/precision/recall/AUC on cassava and wheat datasets vs. single-backbone models | Higher architectural complexity (dual-backbone + attention) increases compute cost, working against the "lightweight/mobile-ready" goal that MobileNetV2 alone is meant to solve |

## Gap Analysis

Across the surveyed literature, three consistent gaps emerge:

1. **Explainability is underused in practice.** Most papers ([2], [3], [4]) report accuracy metrics but do not integrate visual explainability (e.g., Grad-CAM) into a usable, farmer-facing output — even though trust in the prediction matters as much as the prediction itself in a real agricultural setting.
2. **No end-to-end deployable tool.** The reviewed works stop at model evaluation ([1], [5]) or scale up model complexity for marginal accuracy gains ([6]), without addressing how a non-technical user would actually access predictions — no batch processing, no exportable reports, no web interface.
3. **Narrow crop/class scope.** Several papers restrict themselves to a single crop or a handful of diseases ([3], [4]), limiting real-world applicability compared to a broader multi-crop, multi-class system.

## Justification for Chosen Approach

Based on this gap analysis, this project uses **MobileNetV2 as a single, lightweight transfer-learning backbone** (avoiding the compute overhead of dual-backbone approaches like [6]) trained across all **38 classes and 14 crops** of PlantVillage (addressing the narrow-scope gap in [3], [4]), combined with **Grad-CAM explainability** (addressing the trust gap in [2]–[4]) and wrapped in a **full Streamlit deployment** with batch upload, CSV export, and PDF reporting — directly closing the deployment gap left open by [1] and [5].

## References (IEEE Format)

[1] "Plant Leaf Disease Detection using Transfer Learning and Explainable AI," *IEEE Conference Publication*, IEEE Xplore, 2022. [Online]. Available: https://ieeexplore.ieee.org/abstract/document/9946513/

[2] "Plant Disease Detection using CNN and Transfer Learning," *IEEE Conference Publication*, IEEE Xplore, 2021. [Online]. Available: https://ieeexplore.ieee.org/abstract/document/9484957/

[3] "Deep Learning-Based Transfer Learning with MobileNetV2 for Crop Disease Detection," *IEEE Conference Publication*, IEEE Xplore, 2025. [Online]. Available: https://ieeexplore.ieee.org/iel8/10914685/10915211/10915399.pdf

[4] "Transfer Learning Approach for Plant Leaf Disease Detection Using CNN with Pre-Trained Feature Extraction Method MobileNetV2," *IEEE Conference Publication*, IEEE Xplore, 2020. [Online]. Available: https://ieeexplore.ieee.org/document/9331214/

[5] "Plant Leaf Disease Detection using MobileNetV2," *ITM Web of Conferences*, 2025. [Online]. Available: https://www.itm-conferences.org/articles/itmconf/ref/2025/10/itmconf_keis2025_01021/itmconf_keis2025_01021.html

[6] "A robust and light-weight transfer learning-based architecture for accurate detection of leaf diseases across multiple plants using less amount of images (LeafDoc-Net)," *PMC / NCBI*, 2024. [Online]. Available: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10809160/
