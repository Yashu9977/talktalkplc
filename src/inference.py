# inference.py

import pandas as pd
import joblib
from pathlib import Path

def load_model():
    """Load trained model and label encoders"""
    model_path = Path("models/xgboost_churn_model.pkl")
    encoders_path = Path("models/label_encoders.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Train the model first using modeling.py")
    
    if not encoders_path.exists():
        raise FileNotFoundError("Label encoders not found. Train the model first using modeling.py")
    
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    
    return model, label_encoders


def predict_churn(df, model=None, label_encoders=None):
    """
    Predict churn probability for customers
    
    Args:
        df: DataFrame with customer features (must include unique_customer_identifier)
        model: Trained model (optional, will load if not provided)
        label_encoders: Label encoders dict (optional, will load if not provided)
    
    Returns:
        DataFrame with customer_id and churn_probability
    """

    if model is None or label_encoders is None:
        model, label_encoders = load_model()
    
    # Keep customer ID
    customer_ids = df['unique_customer_identifier'].copy()
    
    X = df.drop(['unique_customer_identifier'], axis=1)
    
    if 'is_churned' in X.columns:
        X = X.drop(['is_churned'], axis=1)
    
    cat_cols = ['contract_status', 'tenure_bucket', 'sales_channel', 'technology']
    
    for col in cat_cols:
        if col in X.columns:
            X[col] = label_encoders[col].transform(X[col].astype(str))

    churn_proba = model.predict_proba(X)[:, 1]
    
    results = pd.DataFrame({
        'unique_customer_identifier': customer_ids,
        'churn_probability': churn_proba
    }).sort_values('churn_probability', ascending=False)
    
    return results