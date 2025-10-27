## EDA & Cleaning

Point 1: "During preprocessing, missing BMI values were imputed using the median, since BMI is continuous and mildly right-skewed. Categorical variables (e.g., gender, work_type, smoking_status) were cleaned by trimming whitespace and converting to lowercase to ensure uniform encoding. Rare entries such as “Other” under gender were merged into “Male” to prevent encoding sparsity. Class imbalance was observed (≈95% non-stroke vs 5% stroke), and it will be addressed at the modeling stage using oversampling (SMOTE) and class weights. No extreme outliers were removed from the dataset to preserve clinical variability. The resulting dataset, stroke_final_clean.csv, serves as the modeling input."



Point 2: "A significant class imbalance (≈95% non-stroke vs. 5% stroke) was observed in the dataset. To address this, class balancing strategies were integrated into the model training stage. Specifically, Synthetic Minority Oversampling Technique (SMOTE) was applied within the training folds to generate synthetic minority examples, ensuring balanced learning without altering the test distribution. Additionally, models such as Logistic Regression and XGBoost were configured with internal class weighting (class_weight='balanced' and scale_pos_weight, respectively). These measures are crucial for improving recall and reducing false negatives — an essential requirement in stroke risk prediction.
"


## Modeling & Evalueation
