# main.py

import pandas as pd
import yaml
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pathlib import Path

from feature_engineering import (
    build_ref_dates,
    build_call_features,
    build_talk_hold_features,
    build_payment_friction_features,
    build_contract_tenure_features,
    build_usage_features,
    build_product_tech_speed_features,
    build_churn_target_completed
)

from modeling import train_churn_model
from inference import predict_churn


def load_config(config_path="config.yml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def initialize_spark(config):
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName(config['spark']['app_name']) \
        .master(config['spark']['master']) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    return spark


def load_data(spark, config):

    
    cease_df = spark.read.csv(config['data']['cease'], header=True, inferSchema=True)
    customer_info_df = spark.read.parquet(config['data']['customer_info'])
    call_df = spark.read.csv(config['data']['calls'], header=True, inferSchema=True)
    usage_df = spark.read.parquet(config['data']['usage'])
    
    print(f"Cease data: {cease_df.count()} rows")
    print(f"Customer info: {customer_info_df.count()} rows")
    print(f"Call data: {call_df.count()} rows")
    print(f"Usage data: {usage_df.count()} rows")
    
    return cease_df, customer_info_df, call_df, usage_df


def build_training_features(cease_df, customer_info_df, call_df, usage_df):
    """Build all features using PySpark"""
    
    # Step 1: Build reference dates
    print("\n1. Building reference dates...")
    ref_dates = build_ref_dates(cease_df, customer_info_df)
    
    # Step 2: Build call features
    print("2. Building call features...")
    call_features = build_call_features(call_df, ref_dates)
    
    # Step 3: Build talk/hold time features
    print("3. Building talk and hold time features...")
    talk_hold_features = build_talk_hold_features(call_df, ref_dates)
    
    # Step 4: Build payment friction features
    print("4. Building payment friction features...")
    payment_features = build_payment_friction_features(customer_info_df)
    
    # Step 5: Build contract and tenure features
    print("5. Building contract and tenure features...")
    contract_features = build_contract_tenure_features(customer_info_df)
    
    # Step 6: Build usage features
    print("6. Building usage features...")
    usage_features = build_usage_features(usage_df, ref_dates)
    
    # Step 7: Build product/tech/speed features
    print("7. Building product, technology and speed features...")
    product_features = build_product_tech_speed_features(customer_info_df)
    
    # Step 8: Build churn target
    print("8. Building churn target...")
    churn_target = build_churn_target_completed(cease_df)
    
    # Step 9: Join all features (PySpark)
    print("\n9. Joining all features...")
    final_df = ref_dates.select("unique_customer_identifier")
    
    final_df = final_df.join(call_features, "unique_customer_identifier", "left")
    final_df = final_df.join(talk_hold_features, "unique_customer_identifier", "left")
    final_df = final_df.join(payment_features, "unique_customer_identifier", "left")
    final_df = final_df.join(contract_features, "unique_customer_identifier", "left")
    final_df = final_df.join(usage_features, "unique_customer_identifier", "left")
    final_df = final_df.join(product_features, "unique_customer_identifier", "left")
    final_df = final_df.join(churn_target.select("unique_customer_identifier", "is_churned"), 
                             "unique_customer_identifier", "left")
    
    # Fill missing churn target with 0 (non-churners)
    final_df = final_df.fillna({"is_churned": 0})
    
    print(f"\nFinal feature set: {final_df.count()} rows, {len(final_df.columns)} columns")
    
    return final_df


def build_inference_features(customer_info_df, call_df, usage_df):

    print("\nBuilding reference dates for inference...")
    ref_dates = customer_info_df.groupBy("unique_customer_identifier") \
        .agg(F.max("datevalue").alias("ref_date"))
    
    print("1. Building call features...")
    call_features = build_call_features(call_df, ref_dates)
    
    print("2. Building talk and hold time features...")
    talk_hold_features = build_talk_hold_features(call_df, ref_dates)
    
    print("3. Building payment friction features...")
    payment_features = build_payment_friction_features(customer_info_df)
    
    print("4. Building contract and tenure features...")
    contract_features = build_contract_tenure_features(customer_info_df)
    
    print("5. Building usage features...")
    usage_features = build_usage_features(usage_df, ref_dates)
    
    print("6. Building product, technology and speed features...")
    product_features = build_product_tech_speed_features(customer_info_df)
    
    print("\n7. Joining all features...")
    final_df = ref_dates.select("unique_customer_identifier")
    final_df = final_df.join(call_features, "unique_customer_identifier", "left")
    final_df = final_df.join(talk_hold_features, "unique_customer_identifier", "left")
    final_df = final_df.join(payment_features, "unique_customer_identifier", "left")
    final_df = final_df.join(contract_features, "unique_customer_identifier", "left")
    final_df = final_df.join(usage_features, "unique_customer_identifier", "left")
    final_df = final_df.join(product_features, "unique_customer_identifier", "left")
    
    print(f"\nFinal feature set: {final_df.count()} rows, {len(final_df.columns)} columns")
    
    return final_df


def train_pipeline(spark, config):
    
    cease_df, customer_info_df, call_df, usage_df = load_data(spark, config)
    
    features_spark_df = build_training_features(cease_df, customer_info_df, call_df, usage_df)
    
    features_pdf = features_spark_df.toPandas()
    
    # Save features
    print(f"\nSaving features to {config['output']['features']}...")
    features_pdf.to_parquet(config['output']['features'], index=False)
    print("Features saved successfully!")
    
    # Train model (Pandas)

    model, metrics = train_churn_model(features_pdf)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {config['output']['model']}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return model, metrics


def inference_pipeline(spark, config): 
    
    customer_info_df = spark.read.parquet(config['data']['customer_info'])
    call_df = spark.read.csv(config['data']['calls'], header=True, inferSchema=True)
    usage_df = spark.read.parquet(config['data']['usage'])
    
    print(f"Customer info: {customer_info_df.count()} rows")
    print(f"Call data: {call_df.count()} rows")
    print(f"Usage data: {usage_df.count()} rows")
    
    features_spark_df = build_inference_features(customer_info_df, call_df, usage_df)
    
    features_pdf = features_spark_df.toPandas()
    
    predictions = predict_churn(features_pdf)
    
    # Save predictions
    predictions.to_csv(config['output']['predictions'], index=False)
    print(f"\nPredictions saved to: {config['output']['predictions']}")
    
    return predictions


if __name__ == "__main__": 
    config = load_config("config.yml")
    
    # Initialize Spark
    spark = initialize_spark(config)

    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nChoose mode:")
        print("1. Train model")
        print("2. Run inference")
        choice = input("\nEnter choice (1 or 2): ")
        mode = "train" if choice == "1" else "inference"
    
    try:
        if mode == "train":
            model, metrics = train_pipeline(spark, config)
        elif mode == "inference":
            predictions = inference_pipeline(spark, config)
        else:
            print("Invalid mode. Use 'train' or 'inference'")
    
    finally:
        spark.stop()
        print("\nSpark session stopped.")