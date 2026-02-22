import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import joblib
from pathlib import Path

def train_churn_model(df):
    """
    Train XGBoost churn model with MLflow tracking
    
    Args:
        df: DataFrame with features and is_churned target
    """
    Path("models").mkdir(exist_ok=True)

    mlflow.set_experiment("telecom_churn")

    X = df.drop(['unique_customer_identifier', 'is_churned'], axis=1)
    y = df['is_churned']
    
    cat_cols = ['contract_status', 'tenure_bucket', 'sales_channel', 'technology']
    label_encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    with mlflow.start_run():
        # Model parameters
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),
            'random_state': 42
        }
        
        mlflow.log_params(params)
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        mlflow.log_metrics(metrics)
        
        print("\nModel Performance:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
    
        mlflow.xgboost.log_model(model, "model")
        
        joblib.dump(model, "models/xgboost_churn_model.pkl")
        joblib.dump(label_encoders, "models/label_encoders.pkl")
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv("models/feature_importance.csv", index=False)
        mlflow.log_artifact("models/feature_importance.csv")
        
        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))
        print(f"\nModel saved to: models/xgboost_churn_model.pkl")
        
    return model, metrics


# Usage
if __name__ == "__main__":
    df = pd.read_parquet("")
    model, metrics = train_churn_model(df)