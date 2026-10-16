import os
import json
import sys
import pandas as pd
import numpy as np
from typing import TypedDict, List, Any
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# 1. Environment Configurations

os.environ["GROQ_API_KEY"] = dbutils.secrets.get(scope="tokens", key="groq_api_key")

sys.path.append(os.path.abspath('.'))
sys.path.append('/Workspace/Git/sce/simulation/data')
sys.path.append('/Workspace/Git/sce/simulation/model')

import data_transform
import classification_model

data_transform.spark = spark

# --- included model object slots & context trackers natively inside state ---
class TriangulatedState(TypedDict):
    full_test_df: pd.DataFrame
    audit_rows_df: pd.DataFrame
    full_data_df: pd.DataFrame
    X_columns: List[str]
    desired_threshold: float
    full_forecast_df: pd.DataFrame
    # Explicitly holding pipeline references across sequential states
    lr_model: Any
    rf_model: Any
    xgb_model: Any

class GridPredictionSchema(BaseModel):
    predictions: List[int] = Field(description="List of binary predictions (0 or 1) matching the input payload order.")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0, api_key=GROQ_API_KEY)
structured_llm = llm.with_structured_output(GridPredictionSchema)

# 2. Node 1: testing label inference pipeline

def node_executed_all_models_triangulation(state: TriangulatedState) -> dict:
    spark_df = data_transform._read_data()
    X_train, y_train, X_test, y_test, pdf, X_columns = data_transform._transform_data(spark_df)
    
    desired_threshold = 0.5 
    full_df, model_metrics, lr_pipe, rf_pipe, xgb_pipe = classification_model._train_triangulation_models(
        X_train, y_train, X_test, y_test, pdf, desired_threshold
    )
    
    full_test_df = full_df[(full_df["dataset_split"] == "test")].copy()
    full_test_df["groq_pred"] = full_test_df["rf_pred"]
    
    disagreement_mask = (full_test_df["lr_pred"] != full_test_df["xgb_pred"]) | \
                        (full_test_df["rf_pred"] != full_test_df["xgb_pred"])
    audit_rows = full_test_df[disagreement_mask].copy()
    
    print(f"Token Optimization: Filtered {len(full_test_df)} down to {len(audit_rows)} conflict rows for Groq verification.")
    
    return {
        "full_test_df": full_test_df, 
        "audit_rows_df": audit_rows, 
        "lr_model": lr_pipe, 
        "rf_model": rf_pipe, 
        "xgb_model": xgb_pipe, 
        "full_data_df": pdf, 
        "X_columns": X_columns, 
        "desired_threshold": desired_threshold
    }


# 3. Node 2: Grow test data arbitration/audit

def node_targeted_groq_audit(state: TriangulatedState) -> dict:
    audit_df = state["audit_rows_df"].copy()
    full_test_df = state["full_test_df"].copy()
    
    if audit_df.empty:
        print("Skipping Groq audit: no conflicting test rows to audit across different models.")
        # --- FIXED 2: Preserved all incoming pipeline objects so they propagate to Node 3 ---
        return {**state, "full_test_df": full_test_df}
        
    skill_file_path = "/Workspace/Git/sce/simulation/ai_groq/databricks_groq_skill.md"
    with open(skill_file_path, "r", encoding="utf-8") as f:
        ai_skill_markdown = f.read()
        
    ai_predictions = []
    chunk_size = 50
    for i in range(0, len(audit_df), chunk_size):
        chunk = audit_df.iloc[i:i+chunk_size]
        compact_payload = chunk[["zip_code", "temperature_f", "avg_kwh", "wildfire_tier", "lr_pred", "rf_pred", "xgb_pred"]].to_json(orient="records")
        prompt = f"{ai_skill_markdown}\n## Model Conflict Logs To Arbitrate:\n{compact_payload}"
        
        try:
            result_schema = structured_llm.invoke(prompt)
            chunk_predictions = result_schema.predictions
            if len(chunk_predictions) != len(chunk):
                chunk_predictions = (chunk_predictions + [0] * len(chunk))[:len(chunk)]
            ai_predictions.extend(chunk_predictions)
        except Exception as e:
            ai_predictions.extend( * len(chunk))
            
    if len(ai_predictions) != len(audit_df):
        ai_predictions = (ai_predictions + [0] * len(audit_df))[:len(audit_df)]
        
    audit_df["groq_pred"] = ai_predictions
    
    full_test_df["groq_pred"] = full_test_df["groq_pred"].astype("int64")
    audit_df["groq_pred"] = audit_df["groq_pred"].astype("int64")
    
    full_test_df.set_index(["zip_code", "reporting_date"], inplace=True)
    audit_df.set_index(["zip_code", "reporting_date"], inplace=True)
    full_test_df.update(audit_df[["groq_pred"]])
    full_test_df.reset_index(inplace=True)
    
    # --- FIXED 2: Using Python kwargs expansion to append full_test_df changes while keeping old pipeline references intact ---
    return {**state, "full_test_df": full_test_df}


# 4. Node 3: groq forecast arbitration/audit

def node_targeted_groq_fcst_audit(state: TriangulatedState) -> dict:
    lr_model = state["lr_model"]
    rf_model = state["rf_model"]
    xgb_model = state["xgb_model"] # --- FIXED 3: Corrected variable reassignment overwrite typo ---
    X_columns = state["X_columns"]
    pdf = state["full_data_df"].copy()
    desired_threshold = state["desired_threshold"]
    
    # --- FIXED 4: Added Identity Keys (zip_code, reporting_date) so spatial index updates work ---
    identity_cols = ["zip_code", "reporting_date"]
    forecast_df = pdf[pdf["dataset_split"] == "forecast"][identity_cols + X_columns].copy().reset_index(drop=True)
    
    # Generate binary future predictions
    forecast_df["lr_pred"] = (lr_model.predict_proba(forecast_df[X_columns])[:, 1] >= desired_threshold).astype(int)
    forecast_df["rf_pred"] = (rf_model.predict_proba(forecast_df[X_columns])[:, 1] >= desired_threshold).astype(int)
    forecast_df["xgb_pred"] = (xgb_model.predict_proba(forecast_df[X_columns])[:, 1] >= desired_threshold).astype(int)
    
    forecast_df["groq_pred"] = forecast_df["rf_pred"]
    
    conflict_mask = (forecast_df["lr_pred"] != forecast_df["xgb_pred"]) | \
                    (forecast_df["rf_pred"] != forecast_df["xgb_pred"])
    forecast_audit_rows = forecast_df[conflict_mask].copy()
    print(f"Token Optimization: Found {len(forecast_audit_rows)} uncertain future rows out of {len(forecast_df)} for Groq processing.")
    
    if not forecast_audit_rows.empty:
        skill_file_path = "/Workspace/Git/sce/simulation/ai_groq/databricks_groq_skill.md"
        with open(skill_file_path, "r", encoding="utf-8") as f:
            ai_skill_markdown = f.read()
            
        groq_forecast_predictions = []
        chunk_size = 50
        for i in range(0, len(forecast_audit_rows), chunk_size):
            chunk = forecast_audit_rows.iloc[i:i+chunk_size]
            compact_payload = chunk[["zip_code", "temperature_f", "avg_kwh", "wildfire_tier", "lr_pred", "rf_pred", "xgb_pred"]].to_json(orient="records")
            
            prompt = f"""
            {ai_skill_markdown}
            ## TARGET FUTURE FORECAST LOGS TO ARBITRATE:
            {compact_payload}
            """
            try:
                result_schema = structured_llm.invoke(prompt)
                chunk_predictions = result_schema.predictions
                if len(chunk_predictions) != len(chunk):
                    chunk_predictions = (chunk_predictions + [0] * len(chunk))[:len(chunk)]
                groq_forecast_predictions.extend(chunk_predictions)
            except Exception as e:
                groq_forecast_predictions.extend( [0] * len(chunk))
                
        forecast_audit_rows["groq_pred"] = groq_forecast_predictions
        
        forecast_df["groq_pred"] = forecast_df["groq_pred"].astype("int64")
        forecast_audit_rows["groq_pred"] = forecast_audit_rows["groq_pred"].astype("int64")
        
        forecast_df.set_index(["zip_code", "reporting_date"], inplace=True)
        forecast_audit_rows.set_index(["zip_code", "reporting_date"], inplace=True)
        forecast_df.update(forecast_audit_rows[["groq_pred"]])
        forecast_df.reset_index(inplace=True)
        
    return {**state, "full_forecast_df": forecast_df}

# 5. assemble graph workflow

def main():

    workflow = StateGraph(TriangulatedState)
    workflow.add_node("triangulated_models", node_executed_all_models_triangulation)
    workflow.add_node("audit_conflicts", node_targeted_groq_audit)
    workflow.add_node("audit_fcst_conflicts", node_targeted_groq_fcst_audit)

    workflow.add_edge(START, "triangulated_models")
    workflow.add_edge("triangulated_models", "audit_conflicts")
    workflow.add_edge("audit_conflicts", "audit_fcst_conflicts")
    workflow.add_edge("audit_fcst_conflicts", END)

    production_agent = workflow.compile()

    # Invoke the structured computational graph layout
    output_state = production_agent.invoke({})
    
    final_test_df = output_state["full_test_df"]

    spark_test_df = spark.createDataFrame(final_test_df)
    spark_test_df.write.format("delta").mode("overwrite").saveAsTable("sce_db.default.fact_grid_vulnerabilities_by_zip_test")

    final_fcst_df = output_state["full_forecast_df"]

    spark_fcst_df = spark.createDataFrame(final_fcst_df)
    spark_fcst_df.write.format("delta").mode("overwrite").saveAsTable("sce_db.default.fact_grid_vulnerabilities_by_zip_fcst")

if __name__ == "__main__":
    main()