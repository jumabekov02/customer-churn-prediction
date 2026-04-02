# Customer Churn Prediction

**End-to-end machine learning pipeline predicting customer churn with 89% accuracy and fully interpretable results**

## Project Overview

Built a complete ML pipeline to predict binary customer outcomes (churn/retention) from 9,000 records with mixed numerical and categorical features. Focused on both predictive performance and model interpretability for stakeholder trust.

## Key Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 89% |
| **F1-Score** | 0.88 |
| **ROC-AUC** | 0.96 |
| **Precision** | 89% |
| **Recall** | 86% |

- **Top churn drivers identified:** `f3_f4_diff`, `feature_1`, `feature_2` (via SHAP)
- **4 engineered features** outperformed raw variables
- **Zero data leakage** with robust train/test separation

## Tech Stack

- **Languages:** Python
- **Libraries:** scikit-learn, pandas, NumPy, SHAP, matplotlib, seaborn
- **Techniques:** Random Forest, Gradient Boosting, feature engineering, hyperparameter tuning (RandomizedSearchCV), SHAP interpretability

## What I Did

### 1. Data Exploration & Preprocessing
- Analyzed 9,000 samples with 10 features (8 numerical, 2 categorical)
- Handled 4.4% missing data with median imputation (robust to outliers)
- Built preprocessing pipelines with StandardScaler and OneHotEncoder
- Enforced strict train/test split with stratification to prevent leakage

### 2. Feature Engineering
- Created 4 predictive features guided by EDA insights:
  - `f1_f2_ratio` — relative relationship between key variables
  - `f3_f4_diff` — contrast between correlated measurements
  - `numeric_sum` & `numeric_mean` — aggregated magnitude indicators
- Validated engineered features through distribution analysis vs. target

### 3. Modeling & Evaluation
- Established baseline with Logistic Regression (87.7% accuracy)
- Trained 3 ensemble models: Random Forest, Gradient Boosting, AdaBoost
- Selected Random Forest for hyperparameter tuning (best F1 balance)
- Optimized with 5-fold cross-validation using RandomizedSearchCV

### 4. Model Interpretation
- Extracted feature importance to identify key predictors
- Applied SHAP analysis for transparent, instance-level explanations
- Confirmed engineered features (`f3_f4_diff`, `f1_f2_ratio`) as top contributors

## How to Run

```bash
# Clone repository
git clone https://github.com/[your-username]/customer-churn-prediction.git

# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook ML_Final_Project.ipynb
