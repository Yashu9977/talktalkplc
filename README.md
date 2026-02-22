# UK Telecoms Customer Churn Prediction

A machine learning solution to predict customer churn for UK Telecoms LTD, enabling proactive retention strategies by identifying high-risk customers before they leave.

## 📋 Table of Contents
- [Business Problem](#business-problem)
- [Solution Overview](#solution-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Features](#features)
- [Model Performance](#model-performance)
- [Technical Stack](#technical-stack)

## 🎯 Business Problem

UK Telecoms LTD wants to prioritize customer retention by identifying customers likely to churn (place a cease). The goal is to enable the retention team to focus their efforts on high-risk customers through proactive outreach.

**Key Objectives:**
- Predict which customers are most likely to churn
- Provide a prioritized list for the retention team
- Understand key drivers of customer churn
- Enable data-driven retention strategies

## 🔧 Solution Overview

This project implements a **two-stage approach**:

### Stage 1: Understanding the Patterns
Comprehensive exploratory data analysis to answer critical business questions:
- **Who is leaving?** (customer demographics, contract status, tenure)
- **Why are they leaving?** (stated reasons, service quality issues)
- **How are they behaving before they leave?** (call patterns, usage drops, payment issues)
- **When are they leaving?** (contract lifecycle, seasonal patterns)

### Stage 2: Model Building
- **Feature Engineering:** PySpark for scalable feature extraction from large datasets
- **Modeling:** XGBoost classifier trained on engineered features
- **Inference:** Pandas-based prediction pipeline for scoring customers
- **MLflow Tracking:** Complete experiment tracking and model versioning


## 🚀 Installation

### Prerequisites
- Python 3.8+
- Java 8+ (for PySpark)

### Setup

1. **Clone the repository:**

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Java 8+ (for PySpark)

### Setup

1. **Clone the repository:**
```bash
git clone 
cd churn-prediction
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure data paths:**
Edit `config.yml` to point to your data files:
```yaml
data:
  cease: "data/cease_data.csv"
  customer_info: "data/customer_info.parquet"
  calls: "data/call_information.csv"
  usage: "data/usage_data.parquet"
```

## 💻 Usage

### Training the Model
```bash
# Run full training pipeline
python main.py train
```

This will:
1. Load data using PySpark
2. Engineer features from all data sources
3. Convert to Pandas for modeling
4. Train XGBoost model with class imbalance handling
5. Log metrics and artifacts to MLflow
6. Save model to `models/` directory
