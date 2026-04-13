# 📉 Customer Churn Prediction
> Predicting telecom customer churn using machine learning — from raw data to a tuned, production-ready classifier.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nkosimate/CustomerChurnPrediction/blob/main/Churn.ipynb)

---

## Overview

A telecom company loses revenue every time a customer walks. This project builds an end-to-end ML pipeline to identify customers likely to churn **before** they do  giving retention teams a targeted, cost-efficient list to act on.

**Dataset:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) · 7,043 customers · 21 features

---

## Pipeline

```
Raw Data → EDA → Feature Engineering → Preprocessing → Baseline Models → Hyperparameter Tuning → Final Model
```

| Phase | What happens |
|---|---|
| **EDA** | Class balance check, univariate/bivariate analysis, correlation heatmap, pairplot |
| **Feature Engineering** | 3 derived features: `HighCallUser`, `DataMismatch`, `UsagePerDollar` |
| **Preprocessing** | StandardScaler on continuous features, binary flags passed through; stratified 80/20 split |
| **Modelling** | Logistic Regression, Decision Tree, Random Forest, Gradient Boosting — all with `class_weight='balanced'` |
| **Tuning** | RandomizedSearchCV (broad) → GridSearchCV (narrow) on Gradient Boosting |

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| Random Forest | — | — | — | ✦ | **Best AUC** |
| **Gradient Boosting** ✅ | — | — | — | **Best** | — |

> **Selected model: Gradient Boosting** — chosen for highest F1-score and precision, minimising costly false positives. Hyperparameter tuning improved precision by ~10% with a minor, intentional trade-off in ROC-AUC.

---

## Key Findings

- **~14.5% churn rate** — moderate class imbalance handled via balanced class weights
- Customers with **≥4 service calls** show a sharp cliff in churn probability
- **High monthly charges** and **no contract renewal** are strong churn predictors
- Customers on a data plan with **zero data usage** are a high-risk segment

---

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn`

---

## Quickstart

```bash
git clone https://github.com/nkosimate/CustomerChurnPrediction.git
cd CustomerChurnPrediction
pip install -r requirements.txt
jupyter notebook Churn.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/github/nkosimate/CustomerChurnPrediction/blob/main/Churn.ipynb).

---

## Project Structure

```
CustomerChurnPrediction/
├── Churn.ipynb          # Full notebook — EDA through tuned model
├── telecom_churn.csv    # Dataset
└── README.md
```
