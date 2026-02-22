import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

spark = (
    SparkSession.builder
    .appName("Churn-Analysis")
    .master("local[*]")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate()
)


def build_ref_dates(cease_df, customer_info_df):
    churn_ref = cease_df.filter(F.col("cease_placed_date").isNotNull()) \
            .groupBy("unique_customer_identifier") \
            .agg(F.min("cease_placed_date").alias("ref_date")) \
            .withColumn("churn_group", F.lit("Churner"))

    nonchurn_ref = customer_info_df.groupBy("unique_customer_identifier") \
        .agg(F.max("datevalue").alias("ref_date")) \
        .join(churn_ref.select("unique_customer_identifier"), on="unique_customer_identifier", how="left_anti") \
        .withColumn("churn_group", F.lit("Non-churner"))

    return churn_ref.unionByName(nonchurn_ref)


def build_call_features(call_df, ref_dates):
    
    call_features = ref_dates.select("unique_customer_identifier", "ref_date").join(
        call_df.select("unique_customer_identifier", "event_date", "call_type"),
        on="unique_customer_identifier",
        how="left"
    ).withColumn(
        "days_before", F.datediff(F.col("ref_date"), F.col("event_date"))
    )

    call_features_90 = call_features.filter((F.col("days_before") >= 0) & (F.col("days_before") <= 90))

    is_loyalty = F.col("call_type") == F.lit("Loyalty")
    is_tech    = F.col("call_type") == F.lit("Tech")
    is_csb     = F.col("call_type") == F.lit("CS&B")
    is_fin     = F.col("call_type") == F.lit("Customer Finance")

    feats = call_features_90.groupBy("unique_customer_identifier").agg(

        F.sum(F.when(F.col("days_before") <= 30, 1).otherwise(0)).alias("calls_total_30d"),
        F.sum(F.when(F.col("days_before") <= 60, 1).otherwise(0)).alias("calls_total_60d"),
        F.sum(F.when(F.col("days_before") <= 90, 1).otherwise(0)).alias("calls_total_90d"),

        F.sum(F.when((F.col("days_before") <= 30) & is_loyalty, 1).otherwise(0)).alias("calls_loyalty_30d"),
        F.sum(F.when((F.col("days_before") <= 90) & is_loyalty, 1).otherwise(0)).alias("calls_loyalty_90d"),

        F.sum(F.when((F.col("days_before") <= 30) & is_tech, 1).otherwise(0)).alias("calls_tech_30d"),
        F.sum(F.when((F.col("days_before") <= 90) & is_tech, 1).otherwise(0)).alias("calls_tech_90d"),

        F.sum(F.when((F.col("days_before") <= 30) & is_csb, 1).otherwise(0)).alias("calls_csb_30d"),
        F.sum(F.when((F.col("days_before") <= 90) & is_csb, 1).otherwise(0)).alias("calls_csb_90d"),

        F.sum(F.when((F.col("days_before") <= 30) & is_fin, 1).otherwise(0)).alias("calls_finance_30d"),
        F.sum(F.when((F.col("days_before") <= 90) & is_fin, 1).otherwise(0)).alias("calls_finance_90d"),

        F.min("days_before").alias("days_since_last_call"),

        F.countDistinct("call_type").alias("call_type_diversity_90d")
    )

    base = ref_dates.select("unique_customer_identifier").distinct()
    feats = base.join(feats, on="unique_customer_identifier", how="left").fillna({
        "calls_total_30d": 0, "calls_total_60d": 0, "calls_total_90d": 0,
        "calls_loyalty_30d": 0, "calls_loyalty_90d": 0,
        "calls_tech_30d": 0, "calls_tech_90d": 0,
        "calls_csb_30d": 0, "calls_csb_90d": 0,
        "calls_finance_30d": 0, "calls_finance_90d": 0,
        "call_type_diversity_90d": 0
    }).withColumn(
        "days_since_last_call", F.coalesce(F.col("days_since_last_call"), F.lit(91))
    ).withColumn(
        "repeat_caller_flag_30d", F.when(F.col("calls_total_30d") >= 3, 1).otherwise(0)
    )

    return feats


def build_talk_hold_features(call_df, ref_dates):
    
    calls = call_df \
        .withColumn("event_date", F.to_date("event_date")) \
        .withColumn("talk_time_seconds", (F.col("talk_time_seconds").cast("double"))/ 60) \
        .withColumn("hold_time_seconds", (F.col("hold_time_seconds").cast("double"))/60)

    talk_talk = ref_dates.select("unique_customer_identifier", "ref_date") \
        .join(
            calls.select("unique_customer_identifier", "event_date",
                         "talk_time_seconds", "hold_time_seconds"),
            on="unique_customer_identifier",
            how="left"
        ) \
        .withColumn(
            "days_before", F.datediff(F.col("ref_date"), F.col("event_date"))
        )

    talk_talk_90 = talk_talk.filter(
        (F.col("days_before") >= 0) &
        (F.col("days_before") <= 90)
    )

    feats = talk_talk_90.groupBy("unique_customer_identifier").agg(

        F.sum(F.when(F.col("days_before") <= 30,
                     F.col("talk_time_seconds")).otherwise(0)
              ).alias("talk_time_total_30d"),

        F.sum(F.when(F.col("days_before") <= 90,
                     F.col("talk_time_seconds")).otherwise(0)
              ).alias("talk_time_total_90d"),

        F.sum(F.when(F.col("days_before") <= 30,
                     F.col("hold_time_seconds")).otherwise(0)
              ).alias("hold_time_total_30d"),

        F.sum(F.when(F.col("days_before") <= 90,
                     F.col("hold_time_seconds")).otherwise(0)
              ).alias("hold_time_total_90d"),

        F.sum(F.when(F.col("days_before") <= 90, 1).otherwise(0)
              ).alias("calls_total_90d")
    )

    base = ref_dates.select("unique_customer_identifier").distinct()

    feats = base.join(feats, on="unique_customer_identifier", how="left") \
        .fillna({
            "talk_time_total_30d": 0.0,
            "talk_time_total_90d": 0.0,
            "hold_time_total_30d": 0.0,
            "hold_time_total_90d": 0.0,
            "calls_total_90d": 0
        })

    feats = feats \
        .withColumn(
            "avg_talk_time_per_call_90d",
            F.when(F.col("calls_total_90d") > 0,
                   F.col("talk_time_total_90d") / F.col("calls_total_90d"))
             .otherwise(0.0)
        ) \
        .withColumn(
            "hold_ratio_90d",
            F.when(F.col("talk_time_total_90d") > 0,
                   F.col("hold_time_total_90d") / F.col("talk_time_total_90d"))
             .otherwise(0.0)
        )

    return feats.drop('calls_total_90d')


def build_payment_friction_features(customer_info_df):
    cust = customer_info_df.withColumn("datevalue", F.to_date("datevalue"))

    w = Window.partitionBy("unique_customer_identifier").orderBy(F.col("datevalue").desc())

    feats = cust.withColumn("rn", F.row_number().over(w)) \
        .filter(F.col("rn") == 1) \
        .select(
            "unique_customer_identifier",
            F.coalesce(F.col("contract_dd_cancels").cast("int"), F.lit(0)).alias("contract_dd_cancels"),
            F.coalesce(F.col("dd_cancel_60_day").cast("int"), F.lit(0)).alias("dd_cancel_60_day")
        ) \
        .withColumn("any_contract_dd_cancel", F.when(F.col("contract_dd_cancels") > 0, 1).otherwise(0)) \
        .withColumn("any_dd_cancel_60d", F.when(F.col("dd_cancel_60_day") > 0, 1).otherwise(0))

    return feats

def build_contract_tenure_features(customer_info_df):
    
    cust = customer_info_df.withColumn("datevalue", F.to_date("datevalue"))
    w = Window.partitionBy("unique_customer_identifier").orderBy(F.col("datevalue").desc())

    df = (cust.withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .select(
            "unique_customer_identifier",
            F.coalesce(F.col("tenure_days").cast("int"), F.lit(0)).alias("tenure_days"),
            F.coalesce(F.col("ooc_days").cast("int"), F.lit(0)).alias("ooc_days"),
            F.coalesce(F.col("contract_status"), F.lit("Unknown")).alias("contract_status")
        )
        .withColumn("is_out_of_contract", F.when(F.col("ooc_days") > 0, 1).otherwise(0))
        .withColumn("is_near_ooc", F.when((F.col("ooc_days") >= -60) & (F.col("ooc_days") <= 0), 1).otherwise(0))
        .withColumn(
            "tenure_bucket",
            F.when(F.col("tenure_days") < 30,  "0–1 month")
             .when(F.col("tenure_days") < 90,  "1–3 months")
             .when(F.col("tenure_days") < 180, "3–6 months")
             .when(F.col("tenure_days") < 365, "6–12 months")
             .when(F.col("tenure_days") < 540, "12–18 months")
             .when(F.col("tenure_days") < 730, "18–24 months")
             .otherwise("24+ months")
        )
    )

    return df.drop("tenure_days")



def build_usage_features(usage_df, ref_dates):

    usage = usage_df.withColumn("calendar_date", F.to_date("calendar_date")) \
        .withColumn("usage_download_mbs", F.col("usage_download_mbs").cast("double")) \
        .withColumn("usage_upload_mbs", F.col("usage_upload_mbs").cast("double"))

    usage_data = ref_dates.select("unique_customer_identifier", "ref_date").join(
        usage.select("unique_customer_identifier", "calendar_date", "usage_download_mbs", "usage_upload_mbs"),
        on="unique_customer_identifier",
        how="left"
    ).withColumn(
        "days_before", F.datediff(F.col("ref_date"), F.col("calendar_date"))
    )

    usage_data_90 = usage_data.filter((F.col("days_before") >= 0) & (F.col("days_before") <= 60))

    feats = usage_data_90.groupBy("unique_customer_identifier").agg(

        F.sum(F.when(F.col("days_before") <= 30, F.col("usage_download_mbs")).otherwise(0.0)).alias("download_total_30d"),
        F.sum(F.when(F.col("days_before") <= 60, F.col("usage_download_mbs")).otherwise(0.0)).alias("download_total_60d"),

        F.sum(F.when(F.col("days_before") <= 30, F.col("usage_upload_mbs")).otherwise(0.0)).alias("upload_total_30d"),
        F.sum(F.when(F.col("days_before") <= 60, F.col("usage_upload_mbs")).otherwise(0.0)).alias("upload_total_60d"),

        F.min("days_before").alias("days_since_last_usage"),

        F.sum(F.when(F.col("days_before") <= 27, F.col("usage_download_mbs")).otherwise(0.0)).alias("download_last_28d"),
        F.sum(F.when((F.col("days_before") >= 28) & (F.col("days_before") <= 55),
                     F.col("usage_download_mbs")).otherwise(0.0)).alias("download_prev_28d")
    )

    base = ref_dates.select("unique_customer_identifier").distinct()

    feats = base.join(feats, on="unique_customer_identifier", how="left").fillna({
        "download_total_30d": 0.0,
        "download_total_60d": 0.0,
        "upload_total_30d": 0.0,
        "upload_total_60d": 0.0,
        "download_last_28d": 0.0,
        "download_prev_28d": 0.0
    }).withColumn(

        "days_since_last_usage", F.coalesce(F.col("days_since_last_usage"), F.lit(60))
    )

    feats = feats.withColumn(
        "usage_drop_pct_28d",
        F.when(F.col("download_prev_28d") > 0,
               F.round((F.col("download_last_28d") - F.col("download_prev_28d")) / F.col("download_prev_28d") * 100, 2)
        ).otherwise(None)
    ).drop("download_last_28d", "download_prev_28d")

    return feats.select('unique_customer_identifier','usage_drop_pct_28d')


def build_product_tech_speed_features(customer_info_df):
    
    cust = customer_info_df.withColumn("datevalue", F.to_date("datevalue"))

    w = Window.partitionBy("unique_customer_identifier").orderBy(F.col("datevalue").desc())

    df = cust.withColumn("rn", F.row_number().over(w)) \
        .filter(F.col("rn") == 1) \
        .select(
            "unique_customer_identifier",
            F.coalesce(F.col("sales_channel"), F.lit("Unknown")).alias("sales_channel"),
            F.coalesce(F.col("technology"), F.lit("Unknown")).alias("technology"),
            F.col("speed").cast("double").alias("speed"),
            F.col("line_speed").cast("double").alias("line_speed")
        )

    df = df.withColumn(
        "speed_gap",
        F.when(F.col("speed").isNotNull() & F.col("line_speed").isNotNull(),
               F.col("speed") - F.col("line_speed")
        ).otherwise(None)
    ).withColumn(
        "speed_gap_pct",
        F.when((F.col("speed") > 0) & F.col("line_speed").isNotNull(),
               (F.col("speed") - F.col("line_speed")) / F.col("speed")
        ).otherwise(None)
    )

    return df


def build_churn_target_completed(cease_df):
    
    target = cease_df.filter(F.col("cease_completed_date")!='null') \
        .groupBy("unique_customer_identifier") \
        .agg(F.min("cease_completed_date").alias("ref_date")) \
        .withColumn("is_churned", F.lit(1))

    return target
