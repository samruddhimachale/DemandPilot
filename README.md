# DemandPilot: Retail Demand and Inventory Forecasting

An end-to-end **MLOps project for retail demand forecasting** designed to help businesses reduce stock-outs and overstocking through machine learning, automated pipelines, monitoring, explainability, and reproducible model deployment.

---

## 📌 Project Overview

Retail businesses need accurate demand forecasts to maintain the right inventory levels.

Overestimating demand can lead to:

- Overstocking
- Increased storage costs
- Product wastage
- Capital being tied up in inventory

Underestimating demand can lead to:

- Stock-outs
- Lost sales
- Poor customer experience

**DemandPilot** addresses this problem using historical retail sales data and an XGBoost-based forecasting model supported by a complete MLOps workflow.

The project covers the complete lifecycle:

**Data → Versioning → Preprocessing → Feature Engineering → Model Training → Experiment Tracking → Model Selection → Explainability → API → Docker → CI → Monitoring → Drift Detection**

---

## 🎯 Objectives

- Forecast retail product demand using historical sales data.
- Build a reproducible ML pipeline.
- Version datasets and ML artifacts using DVC.
- Track experiments using MLflow.
- Compare a baseline model with XGBoost.
- Deploy the trained model through FastAPI.
- Containerize the API using Docker.
- Automate testing using GitHub Actions.
- Monitor model performance.
- Detect data drift.
- Explain model predictions using SHAP.
- Document responsible AI considerations.

---

# 🏗️ Architecture

```text
                         ┌─────────────────────┐
                         │   M5 Retail Dataset │
                         │  Sales / Calendar / │
                         │       Prices        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       DVC           │
                         │ Data Versioning     │
                         │       + S3          │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Preprocessing     │
                         │  Data Transformation│
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Feature Engineering│
                         │ Lags / Rolling /    │
                         │ Calendar / Price    │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                ┌─────────────────┐   ┌─────────────────┐
                │ Seasonal Naive  │   │    XGBoost      │
                │    Baseline     │   │ Demand Model    │
                └────────┬────────┘   └────────┬────────┘
                         │                     │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │   Model Comparison  │
                         │      & Selection    │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
             ┌────────────┐  ┌────────────┐  ┌────────────┐
             │  MLflow    │  │   SHAP     │  │ Monitoring │
             │ Experiments│  │Explainability│ │ & Drift    │
             └────────────┘  └────────────┘  └────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     FastAPI         │
                         │     REST API        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       Docker        │
                         │ Containerized API   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    CI Pipeline      │
                         │   GitHub Actions    │
                         └─────────────────────┘
