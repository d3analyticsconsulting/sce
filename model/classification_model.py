import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _train_triangulation_models(X_train, y_train, X_test, y_test, pdf, desired_threshold = 0.5):
    #spark_df = _read_data()
    #X_train, y_train, X_test, y_test, pdf = _transform_data(spark_df)
    
    # --- Classification model pipelines ---
    #   logistic regression
    #lr_pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(solver="liblinear", random_state=42))])
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(class_weight={0: 1, 1:10} # 10x penalty for missing the rare event
                                                                                , max_iter=1000))])

    #   random forest
    #rf_pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(class_weight="balanced", random_state=42))])
    rf_pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(class_weight={0: 1, 1:10} # 10x penalty for missing the rare event
                                                                                    , random_state=42))])

    #   xgboost
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_weight = num_neg / num_pos if num_pos > 0 else 1.0
    #xgb_pipe = Pipeline([("scaler", StandardScaler()), ("xgb", XGBClassifier(scale_pos_weight=scale_weight, eval_metric="logloss", random_state=42))])
    xgb_pipe = Pipeline([("scaler", StandardScaler()), ("xgb", XGBClassifier(scale_pos_weight=scale_weight * 5 # Multiply standard ratio by an aggressive multiplier (e.g., 5x)
                                                                             , eval_metric="logloss", random_state=42))])

    # --- Run & evaluate classification models ---
    #   logistic regression
    lr_pipe.fit(X_train, y_train)
    pdf["lr_pred_prob"] = lr_pipe.predict_proba(pdf[X_test.columns])[:, 1]
    #pdf["lr_pred"] = lr_pipe.predict(pdf[X_test.columns])
    pdf["lr_pred"] = (pdf["lr_pred_prob"] >= desired_threshold).astype(int)
    lr_y_prob = pdf[(pdf["dataset_split"] == "test")]["lr_pred_prob"]
    lr_y_pred = pdf[(pdf["dataset_split"] == "test")]["lr_pred"]
    lr_auc = roc_auc_score(y_test, lr_y_prob)
    lr_cm = confusion_matrix(y_test, lr_y_pred)
    lr_false_negatives = lr_cm[1][0]
    lr_p, lr_r, lr_f1, lr_ = precision_recall_fscore_support(y_test, lr_y_pred, labels=[1], average=None, zero_division=0)
    lr_dict = {"auc": lr_auc, "precision": np.mean(lr_p), "recall": np.mean(lr_r), "f1": np.mean(lr_f1), "fn": lr_false_negatives}

    #   random forest
    rf_pipe.fit(X_train, y_train)
    pdf["rf_pred_prob"] = rf_pipe.predict_proba(pdf[X_test.columns])[:, 1]
    #pdf["rf_pred"] = rf_pipe.predict(pdf[X_test.columns])
    pdf["rf_pred"] = (pdf["rf_pred_prob"] >= desired_threshold).astype(int)
    rf_y_prob = pdf[(pdf["dataset_split"] == "test")]["rf_pred_prob"]
    rf_y_pred = pdf[(pdf["dataset_split"] == "test")]["rf_pred"]
    rf_auc = roc_auc_score(y_test, rf_y_prob)
    rf_cm = confusion_matrix(y_test, rf_y_pred)
    rf_false_negatives = rf_cm[1][0]
    rf_p, rf_r, rf_f1, rf_ = precision_recall_fscore_support(y_test, rf_y_pred, labels=[1], average=None, zero_division=0)
    rf_dict = {"auc": rf_auc, "precision": np.mean(rf_p), "recall": np.mean(rf_r), "f1": np.mean(rf_f1), "fn": rf_false_negatives}

    #   xgboost
    xgb_pipe.fit(X_train, y_train)
    pdf["xgb_pred_prob"] = xgb_pipe.predict_proba(pdf[X_test.columns])[:, 1]
    #pdf["xgb_pred"] = xgb_pipe.predict(pdf[X_test.columns])
    pdf["xgb_pred"] = (pdf["xgb_pred_prob"] >= desired_threshold).astype(int)
    xgb_y_prob = pdf[(pdf["dataset_split"] == "test")]["xgb_pred_prob"]
    xgb_y_pred = pdf[(pdf["dataset_split"] == "test")]["xgb_pred"]
    xgb_auc = roc_auc_score(y_test, xgb_y_prob)
    xgb_cm = confusion_matrix(y_test, xgb_y_pred)
    xgb_false_negatives = xgb_cm[1][0]
    xgb_p, xgb_r, xgb_f1, xgb_ = precision_recall_fscore_support(y_test, xgb_y_pred, labels=[1], average=None, zero_division=0)
    xgb_dict = {"auc": xgb_auc, "precision": np.mean(xgb_p), "recall": np.mean(xgb_r), "f1": np.mean(xgb_f1), "fn": xgb_false_negatives}
    
    # --- Results ---
    model_metrics = {"logistic regression": lr_dict, "random forest": rf_dict, "xgboost": xgb_dict}
    
    return pdf, model_metrics, lr_pipe, rf_pipe, xgb_pipe