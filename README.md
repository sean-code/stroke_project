## 🧠 Stroke Risk Prediction and Key Factor Analysis

This project applies **machine learning** techniques to predict the likelihood of **stroke** based on individual clinical and lifestyle data.  
The aim is to build a predictive system that identifies high-risk individuals early, supporting preventive healthcare and evidence-driven decision-making.

## Stroke Project File Setup


```
stroke_project/
│
├── data/
│   └── raw_dataset.csv # Original Kaggle dataset
|   └── stroke_clean.csv # Cleaned dataset ready for modeling

├── notebooks/
│   └── 01_EDA_and_Cleaning.ipynb
│   └── 02_Modeling_and_Evaluation.ipynb
│   └── 03_SHAP_Interpretation.ipynb
├── reports/
│   └── figures/ / # ROC, PR, confusion-matrix plots
│   └── model_metrics.csv # AUC and precision-recall metrics for each model
└── README.md

```


## ⚙️ Workflow Summary
| Stage | Notebook | Key Tasks |
|--------|-----------|-----------|
| **1. EDA & Cleaning** | `01_EDA_and_Cleaning.ipynb` | Handle missing BMI, normalize categoricals, visualize distributions, save `stroke_clean.csv`. |
| **2. Modeling & Evaluation** | `02_Modeling_and_Evaluation.ipynb` | Build Logistic Regression, Random Forest (+SMOTETomek), XGBoost (+tuning & calibration). Evaluate ROC/PR, F2 scores, threshold trade-offs. |
| **3. Interpretation (SHAP)** | `03_SHAP_Interpretation.ipynb` | Compute SHAP values for final model, plot feature importance and local explanations. |

---

## 📈 Results Snapshot
| Model | Imbalance Handling | ROC-AUC | PR-AUC | Notes |
|--------|--------------------|----------|---------|-------|
| Logistic Regression (balanced) | `class_weight='balanced'` | **0.84** | ~0.40 | Linear, interpretable, best generalization |
| Random Forest + SMOTETomek | Oversampling + cleaning | 0.81 | ~0.33 | Captures nonlinear patterns |
| XGBoost (tuned + calibrated) | `scale_pos_weight` | 0.83 | ~0.38 | Best probability calibration |

