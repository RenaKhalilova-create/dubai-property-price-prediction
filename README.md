# Dubai Property Price Prediction using a Stacking Ensemble

Machine Learning project for automated property valuation (AVM) using Dubai Land Department transaction data.

This repository contains the **modeling component** of our university PropTech project. The objective was to build an accurate and interpretable Automated Valuation Model (AVM) capable of estimating Dubai property prices using historical transaction data.

---

## Project Overview

The model predicts the transaction value of residential properties based on engineered features such as:

* Property characteristics
* Location information
* Historical pricing statistics
* Temporal features
* Infrastructure proximity
* Market indicators

Instead of relying on a single algorithm, the final solution uses a **stacking ensemble** that combines multiple machine learning models to improve prediction accuracy.

---

## Model Architecture

### Base Models

* XGBoost
* LightGBM
* Random Forest

### Meta-Learner

* Ridge Regression

The meta-model is trained using **out-of-fold predictions** generated with **TimeSeriesSplit**, preventing temporal data leakage and providing more reliable validation.

---

## Dataset

The project uses transaction data from the **Dubai Land Department (DLD)** covering approximately three years of real estate sales.

Dataset after preprocessing:

* ~448,000 training samples
* 30 engineered features
* Chronological train/test split
* Target-encoded categorical variables
* Rolling market statistics
* One-hot encoded property types

> **Note:** The original dataset is not included because it is too large and subject to usage restrictions.

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* LightGBM
* Joblib

---

## Model Evaluation

Evaluation was performed on a held-out chronological test set.

Performance metrics:

* **MAPE:** 13.22%
* **R²:** 0.928
* **RMSE:** 788,000 AED

Performance by property type:

| Property Type | MAPE  |
| ------------- | ----- |
| Unit          | 12.5% |
| Villa         | 13.0% |
| Land          | 23.9% |

---

## Repository Structure

```
.
├── models/
│   ├── xgb.pkl
│   ├── lgbm.pkl
│   ├── rf.pkl
│   ├── ridge.pkl
│   └── meta_scaler.pkl
│
├── results/
│   ├── overall_metrics.csv
│   └── property_type_metrics.csv
│
├── train_model.py
├── requirements.txt
└── README.md
```

---

## Features

* Stacking Ensemble Regression
* TimeSeriesSplit cross-validation
* Hyperparameter tuning
* Model persistence with Joblib
* Automated evaluation
* Property-type performance analysis

---

## Future Improvements

* Separate model for land valuation
* Longer historical dataset
* Quantile regression for uncertainty estimation
* Additional geospatial features
* Deployment as a web application

---

## Author

**Rena Khalilova**

Modeling Team — PropTech AVM Project

University of Wollongong in Dubai

CSCI323 – Modern Artificial Intelligence
