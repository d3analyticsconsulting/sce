import pandas as pd
import numpy as np
import pyspark.sql.functions as f
from pyspark.sql.window import Window

#mlflow.set_registry_uri("databricks")
#mlflow.set_experiment("/Users/christopherha7@://gmail.com_")

def _read_data():
    
    TABLE_NAME = "sce_db.default.fact_electricity_synthetic_by_zip"
    
    raw_df = (spark.table(TABLE_NAME)
                .select((f.to_date(f.concat(f.col("year"), f.lit("-"), f.col("month"), f.lit("-01"))).cast("date")).alias("reporting_date")
                        ,"zip_code"
                        ,f.when(f.col("is_coastal") == True, f.lit(1)).otherwise(f.lit(0)).alias("is_coastal")
                        ,"wildfire_tier"
                        ,"temperature_f"
                        ,"avg_kwh"
                        ,"failed"
                        )
    )
    
    window_spec = Window.partitionBy("zip_code").orderBy("reporting_date") 
    
    df = (
        raw_df
            .withColumn("dataset_split", f.when(f.col("reporting_date") < "2026-01-01", f.lit("train"))
                                            .when((f.col("reporting_date") >= "2026-01-01") & (f.col("reporting_date") <= "2026-08-01"), f.lit("test"))
                                            .otherwise(f.lit("forecast"))
                    )
            .withColumn("prev_month_temp", f.lag("temperature_f", 1).over(window_spec))
            .withColumn("prev_month_kwh", f.lag("avg_kwh", 1).over(window_spec))
            .withColumn("prev_month_failed", f.lag("failed", 1).over(window_spec))
                .orderBy(["zip_code", "reporting_date"])
    )

    return df


def _transform_data(spark_df):
    # Bring data down to a Pandas DataFrame
    # spark_df = _read_data()
    pdf = spark_df.toPandas()
    pdf["reporting_date"] = pd.to_datetime(pdf["reporting_date"])

    # Ensure chronological sorting for accurate lag calculations per ZIP code
    pdf = pdf.sort_values(by=["zip_code", "reporting_date"]).reset_index(
        drop=True
    )

    # Drop rows with NaNs caused by the first month's shift
    pdf = pdf.dropna().reset_index(drop=True)

    X_columns = [
        "is_coastal",
        "wildfire_tier",
        "temperature_f",
        "avg_kwh",
        "prev_month_temp",
        "prev_month_kwh",
        "prev_month_failed",
    ]

    y_column = "failed"

    train_mask = pdf["dataset_split"] == "train"
    test_mask = pdf["dataset_split"] == "test"

    X_train = pdf.loc[train_mask, X_columns]
    y_train = pdf.loc[train_mask, y_column]
    X_test = pdf.loc[test_mask, X_columns]
    y_test = pdf.loc[test_mask, y_column]

    return X_train, y_train, X_test, y_test, pdf, X_columns