## 🧠 Stroke Risk Prediction and Key Factor Analysis

This project applies **machine learning** techniques to predict the likelihood of **stroke** based on individual clinical and lifestyle data.  
The aim is to build a predictive system that identifies high-risk individuals early, supporting preventive healthcare and evidence-driven decision-making.

## Stroke Project File Setup


```
stroke_project/
│
├── data/
│   ├── raw_dataset.csv
│   ├── stroke_clean.csv
│
├── notebooks/
│   ├── 01_EDA_and_Cleaning.ipynb
│   ├── 02_Modeling_and_Evaluation.ipynb
│   ├── 03_SHAP_Interpretation.ipynb
│
├── reports/
│   ├── figures/
│   │   ├── class_imbalance.png
│   │   ├── eda_distributions.png
│   │   ├── balancing_summary.png
│   │   ├── roc_imbalanced.png
│   │   ├── pr_imbalanced.png
│   │   ├── roc_balanced.png
│   │   ├── pr_balanced.png
│   │   ├── confusion_logistic.png
│   │   ├── confusion_rf.png
│   │   ├── confusion_xgb.png
│   │   ├── shap_beeswarm_xgb.png
│   │   ├── shap_waterfall_high.png
│   │   ├── shap_waterfall_low.png
│   ├── top_shap_xgb.csv
│   ├── Stroke_Final_Paper.pdf
│   ├── Stroke_Final_Slides.pdf
│
└── README.md


```


## ⚙️ Workflow Summary
| Stage                        | Notebook                           | Key Outputs                       | Description                                                                                                                                                        |
| ---------------------------- | ---------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1. Data Cleaning & EDA**   | `01_EDA_and_Cleaning.ipynb`        | `stroke_clean.csv`                | Missing BMI imputed, categorical normalization, outlier capping, EDA visualizations, correlation heatmaps.                                                         |
| **2. Modeling & Evaluation** | `02_Modeling_and_Evaluation.ipynb` | ROC/PR curves, confusion matrices | Trained Logistic, RF, XGBoost on both imbalanced + balanced data; SMOTE-Tomek, class weights, `scale_pos_weight`, XGB tuning, calibration, threshold optimization. |
| **3. SHAP Interpretability** | `03_SHAP_Interpretation.ipynb`     | Beeswarm, local waterfall plots   | Permutation SHAP to explain global + local XGBoost predictions. Extracted top SHAP features.                                                                       |


---

## Modeling Approach
| Technique            | Applied To          | Purpose                               |
| -------------------- | ------------------- | ------------------------------------- |
| **Class Weights**    | Logistic Regression | Penalize misclassified minority class |
| **SMOTE–Tomek**      | Random Forest       | Oversample + clean class boundary     |
| **Scale Pos Weight** | XGBoost             | Adjust gradients for rare class       |








## 📈 Key Results Snapshot
| Model                            | Training Condition | ROC-AUC  | PR-AUC   | Recall   | F2       |
| -------------------------------- | ------------------ | -------- | -------- | -------- | -------- |
| Logistic Regression              | Imbalanced         | 0.84     | 0.58     | 0.32     | 0.41     |
| Random Forest                    | Imbalanced         | 0.77     | 0.44     | 0.29     | 0.35     |
| XGBoost                          | Imbalanced         | 0.80     | 0.52     | 0.34     | 0.42     |
| **Logistic Regression**          | **Balanced**       | 0.86     | 0.65     | 0.71     | 0.68     |
| **RF + SMOTETomek**              | **Balanced**       | 0.81     | 0.61     | 0.67     | 0.64     |
| **XGBoost (tuned + calibrated)** | **Balanced**       | **0.88** | **0.70** | **0.73** | **0.71** |



## Final Deliverables
| File                      | Description                                                    |
| ------------------------- | -------------------------------------------------------------- |
| `Stroke_Final_Paper.pdf`  | IEEE-format research paper covering full methodology & results |
| `Stroke_Final_Slides.pdf` | Final presentation slide deck with visuals & discussion        |
| SHAP plots                | Used for model interpretability                                |
| ROC/PR curves             | Balanced vs. imbalanced comparison                             |
| Confusion matrices        | Supporting threshold analysis                                  |



