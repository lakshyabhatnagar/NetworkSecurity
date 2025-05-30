# Network Security Project for Phishing Data

## Overview

This project delivers an end-to-end machine learning pipeline for detecting phishing websites using URL-based features. It automates the workflow from data ingestion to model deployment, ensuring robust, scalable, and production-ready network security. The system leverages advanced ML techniques, data validation, experiment tracking, and a FastAPI interface for real-time inference.

---

## Table of Contents

- [Features](#features)
- [Pipeline Stages](#pipeline-stages)
- [Key Metrics](#key-metrics)
- [How to Run](#how-to-run)
- [API Usage](#api-usage)
- [Tech Stack & Toolkits](#tech-stack--toolkits)
- [Contributing](#contributing)

---

## Features

- **Automated Data Ingestion:** Loads and splits data from MongoDB.
- **Data Validation:** Schema checks, column validation, and drift detection.
- **Data Transformation:** KNN-based imputation and feature engineering.
- **Model Training:** Trains, tunes, and selects the best model (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost).
- **Experiment Tracking:** MLflow and DagsHub integration for metrics and artifact logging.
- **Deployment:** FastAPI endpoints for real-time predictions and retraining.
- **Cloud Integration:** AWS S3 syncing for model and artifact management.

---

## Pipeline Stages

### 1. Data Ingestion
- Loads `phisingData.csv` into MongoDB and splits data into train/test sets.

### 2. Data Validation
- Checks data integrity using `schema.yaml`, validates column count, and detects data drift.

### 3. Data Transformation
- Imputes missing values using KNN, transforms features, and serializes the preprocessor.

### 4. Model Training & Evaluation
- Trains multiple classifiers, tunes hyperparameters, and selects the best model based on F1-score. Logs metrics to MLflow.

### 5. Deployment
- FastAPI app (`app.py`) provides endpoints for real-time inference and retraining. Artifacts and models are synced to AWS S3.

---

## Key Metrics

- **F1-score:** >92% on test data
- **Precision:** >91% on test data
- **Recall:** Tracked for all models
- **Overfitting/Underfitting Gap:** <5%
- **Experiment Tracking:** 100% of runs logged in MLflow/DagsHub

---

## How to Run

### 1. Install Dependencies
        pip install -r requirements.txt
### 2. Push Data to MongoDB
        python push_data.py

### 3. Run the Main Pipeline
        python main.py

### 4. Start FastAPI Server
        uvicorn app:app --reload

---

## API Usage

- **Predict Phishing URL:**  
  Send a POST request to `/predict` with URL features to get classification results.

- **Retrain Model:**  
  Send a POST request to `/retrain` to trigger model retraining with the latest data.

---

## Tech Stack & Toolkits

- **Languages:** Python
- **ML Libraries:** scikit-learn, pandas, numpy
- **Experiment Tracking:** MLflow, DagsHub
- **API Framework:** FastAPI
- **Cloud:** AWS S3
- **Data Storage:** MongoDB
- **Other Tools:** BentoML, Docker

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or feature requests.

---


**For any queries or support, please contact [lakshyabhatnagar1@gmail.com].**



