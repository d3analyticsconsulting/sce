import numpy as np
import pandas as pd
import pyspark.sql.functions as f


def _sim_data():
    np.random.seed(42)

    # 1. Expand the Geographic and Temporal Footprint
    zip_codes = [f"90{str(i).zfill(3)}" for i in range(1, 101)]
    years = [2024, 2025, 2026]
    months = list(range(1, 13))

    # Assign static baseline traits to each ZIP code
    zip_profiles = {}
    for z in zip_codes:
        zip_profiles[z] = {
            'base_accounts': np.random.randint(2000, 35000),
            'base_kwh': np.random.uniform(250, 600),
            'is_coastal': np.random.choice([True, False], p=[0.3, 0.7]),
            'wildfire_tier': np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
        }

    records = []

    # 2. Simulate the Continuous Time-Series
    for year in years:
        year_temp_anomaly = 3.5 if year == 2025 else 0.0
        
        for month in months:
            for z in zip_codes:
                prof = zip_profiles[z]
                
                # Weather logic
                if prof['is_coastal']:
                    temp_f = np.random.uniform(62, 74) if month not in [7, 8, 9] else np.random.uniform(72, 82)
                else:
                    temp_f = np.random.uniform(55, 85) if month not in [7, 8, 9] else np.random.uniform(88, 108)
                
                # Apply the climate trend anomaly
                temp_f += year_temp_anomaly + np.random.normal(0, 1.5)
                
                # Load calculation matching your SCE portal metrics
                seasonal_multiplier = 1.45 if month in [7, 8, 9] and not prof['is_coastal'] else 1.1
                total_accounts = int(prof['base_accounts'] * (1 + (year - 2024) * 0.015)) 
                avg_kwh = np.round(prof['base_kwh'] * seasonal_multiplier + (temp_f * 0.4), 3)
                total_kwh = int(total_accounts * avg_kwh)
                
                # 3. Physics-Based Failure Logic
                stress_score = (temp_f * 0.4) + (prof['wildfire_tier'] * 15)
                if stress_score > 55:
                    failed = 1
                    outage_severity = np.random.uniform(0.15, 0.50)
                    total_kwh = int(total_kwh * (1 - outage_severity))
                else:
                    failed = 0
                
                # Gather fields into dictionary records
                records.append({
                    'year': year,
                    'month': month,
                    'zip_code': z,
                    'is_coastal': prof['is_coastal'],
                    'wildfire_tier': prof['wildfire_tier'],
                    'temperature_f': np.round(temp_f, 2),
                    'seasonal_multiplier': seasonal_multiplier,
                    'total_accounts': total_accounts,
                    'avg_kwh': avg_kwh,
                    'total_kwh': total_kwh,
                    'failed': failed
                })

    # Define the definitive column order 
    ordered_columns = [
        'year', 'month', 'zip_code', 
        'is_coastal', 'wildfire_tier', 
        'temperature_f', 'seasonal_multiplier', 
        'total_accounts', 'avg_kwh', 'total_kwh', 
        'failed'
    ]

    # Create DataFrame and explicitly lock in the layout sequence
    df = pd.DataFrame(records)[ordered_columns]

    return df

def _write_to_uc(df):

    spark_df = spark.createDataFrame(df)

    # target unity catalog
    TARGET_CATALOG = "sce_db"  # Update to match your actual catalog name if different
    TARGET_SCHEMA = "default"
    TABLE_NAME = f"{TARGET_CATALOG}.{TARGET_SCHEMA}.fact_electricity_synthetic_by_zip"

    spark.sql(f"DROP TABLE IF EXISTS {TABLE_NAME}")

    spark_df.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(TABLE_NAME)

def main():

    table_path = "sce_db.default.fact_electricity_synthetic_by_zip"
    
    if spark.catalog.tableExists(table_path):

        print(f"Table {table_path} already exists. Skipping code execution.")

    else:
        print(f"Table {table_path} does not exist. Executing code...")

        df = _sim_data()

        _write_to_uc(df)

if __name__ == "__main__":
    main()