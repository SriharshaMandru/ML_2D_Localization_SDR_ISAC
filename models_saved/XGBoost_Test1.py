#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ============================================================
# SAFE CPU SETTINGS
# ============================================================
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = "data/processed/train_dataset.csv"
VAL_FILE   = "data/processed/val_dataset.csv"
TEST_FILE  = "data/processed/test_dataset.csv"

OUT_DIR = "results/rssi_filelevel_all_models"
GRAPH_DIR = os.path.join(OUT_DIR, "graphs")
MODEL_DIR = "models_saved"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DIST_COL = "distance_m"
ANGLE_COL = "angle_deg"
RANDOM_STATE = 42


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    print(f"Loaded {path}: {df.shape}")
    return df


def select_rssi_features(df):
    include_keywords = [
        "frequency_mhz",
        "rssi",
        "power",
        "amp",
    ]

    exclude_keywords = [
        "distance_m",
        "x_true",
        "y_true",
        "x_pred",
        "y_pred",
        "phase",
        "cov",
        "r12",
        "r11",
        "r21",
        "r22",
        "aoa",
        "corr",
        "lag",
        "file",
        "path",
        "name",
        "rep",
        "split",
        "tx_available",
    ]

    features = []

    for col in df.columns:
        low = col.lower()

        if any(k in low for k in exclude_keywords):
            continue

        if any(k in low for k in include_keywords):
            if pd.api.types.is_numeric_dtype(df[col]):
                features.append(col)

    if ANGLE_COL in df.columns and pd.api.types.is_numeric_dtype(df[ANGLE_COL]):
        features.append(ANGLE_COL)

    features = sorted(list(set(features)))

    if len(features) == 0:
        raise ValueError("No RSSI features found.")

    return features


def get_group_cols(df):
    if "rx_file_path" in df.columns:
        return ["rx_file_path"]
    elif "rx_file_name" in df.columns:
        return ["rx_file_name"]
    else:
        return [
            c for c in ["frequency_mhz", "distance_m", "angle_deg", "rep"]
            if c in df.columns
        ]


# ============================================================
# FILE-LEVEL FEATURE AGGREGATION
# ============================================================

def make_file_level_dataset(df, feature_cols):
    group_cols = get_group_cols(df)

    if not group_cols:
        raise ValueError("No grouping column found for file-level aggregation.")

    rows = []

    for key, g in df.groupby(group_cols):
        row = {}

        row["group"] = str(key)
        row[DIST_COL] = g[DIST_COL].iloc[0]

        if ANGLE_COL in g.columns:
            row[ANGLE_COL] = g[ANGLE_COL].iloc[0]

        if "frequency_mhz" in g.columns:
            row["frequency_mhz"] = g["frequency_mhz"].iloc[0]

        if "rep" in g.columns:
            row["rep"] = str(g["rep"].iloc[0])

        row["num_windows"] = len(g)

        for col in feature_cols:
            if col not in g.columns:
                continue

            values = pd.to_numeric(g[col], errors="coerce").dropna()

            if len(values) == 0:
                row[f"{col}_median"] = np.nan
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_min"] = np.nan
                row[f"{col}_max"] = np.nan
            else:
                row[f"{col}_median"] = values.median()
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_std"] = values.std()
                row[f"{col}_min"] = values.min()
                row[f"{col}_max"] = values.max()

        rows.append(row)

    return pd.DataFrame(rows)


def select_file_level_features(file_df):
    exclude = {
        "group",
        DIST_COL,
        "rep",
    }

    features = []

    for col in file_df.columns:
        if col in exclude:
            continue

        if pd.api.types.is_numeric_dtype(file_df[col]):
            features.append(col)

    return features


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():
    models = {}

    models["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=2
        ))
    ])

    models["ExtraTrees"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=2
        ))
    ])

    models["XGBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            n_estimators=800,
            max_depth=3,
            learning_rate=0.02,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=2,
            reg_lambda=3.0,
            reg_alpha=0.2
        ))
    ])

    models["MLP"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE
        ))
    ])

    models["SVM"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVR(
            kernel="rbf",
            C=20,
            epsilon=0.05,
            gamma="scale"
        ))
    ])

    return models


# ============================================================
# EVALUATION
# ============================================================

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n================ {name} ================")
    print(f"MAE  : {mae:.4f} m")
    print(f"RMSE : {rmse:.4f} m")
    print(f"R²   : {r2:.4f}")

    return mae, rmse, r2


def save_true_vs_pred(y_true, y_pred, path, title):
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.75, s=45)

    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))

    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    plt.xlabel("True Distance (m)")
    plt.ylabel("Predicted Distance (m)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()


def save_error_hist(errors, path, title):
    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=20)
    plt.xlabel("Absolute Distance Error (m)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()


def save_model_comparison_plot(results_df, path):
    plt.figure(figsize=(8, 5))
    plt.bar(results_df["model"], results_df["test_mae"])
    plt.xlabel("Model")
    plt.ylabel("Test MAE (m)")
    plt.title("RSSI File-Level Distance Model Comparison")
    plt.xticks(rotation=30)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()


def save_distance_confusion_matrix(y_true, y_pred, path, title):
    y_true_class = np.round(y_true).astype(int)
    y_pred_class = np.round(y_pred).astype(int)

    labels = sorted(list(set(y_true_class) | set(y_pred_class)))

    cm = confusion_matrix(
        y_true_class,
        y_pred_class,
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[f"{x}m" for x in labels]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    plt.title(title)
    plt.xlabel("Predicted Distance Class")
    plt.ylabel("True Distance Class")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n===================================================")
    print(" RSSI FILE-LEVEL DISTANCE ESTIMATION - ALL MODELS")
    print(" RandomForest | ExtraTrees | XGBoost | MLP | SVM")
    print("===================================================\n")

    train_df = read_csv(TRAIN_FILE)
    val_df = read_csv(VAL_FILE)
    test_df = read_csv(TEST_FILE)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if DIST_COL not in df.columns:
            raise ValueError(f"{DIST_COL} missing in {name} dataset.")

    # --------------------------------------------------------
    # Select RSSI features
    # --------------------------------------------------------

    rssi_features = select_rssi_features(train_df)

    print("\nWindow-level RSSI features used for aggregation:")
    for c in rssi_features:
        print(" -", c)

    # --------------------------------------------------------
    # Convert window-level to file-level dataset
    # --------------------------------------------------------

    print("\nCreating file-level aggregated datasets...")

    train_file_df = make_file_level_dataset(train_df, rssi_features)
    val_file_df = make_file_level_dataset(val_df, rssi_features)
    test_file_df = make_file_level_dataset(test_df, rssi_features)

    train_file_df.to_csv(
        os.path.join(OUT_DIR, "train_file_level_distance_dataset.csv"),
        index=False
    )
    val_file_df.to_csv(
        os.path.join(OUT_DIR, "val_file_level_distance_dataset.csv"),
        index=False
    )
    test_file_df.to_csv(
        os.path.join(OUT_DIR, "test_file_level_distance_dataset.csv"),
        index=False
    )

    print(f"Train file-level shape: {train_file_df.shape}")
    print(f"Val file-level shape  : {val_file_df.shape}")
    print(f"Test file-level shape : {test_file_df.shape}")

    # --------------------------------------------------------
    # Select file-level features
    # --------------------------------------------------------

    feature_cols = select_file_level_features(train_file_df)

    print(f"\nTotal file-level features: {len(feature_cols)}")

    with open(os.path.join(OUT_DIR, "selected_file_level_distance_features.txt"), "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    X_train = train_file_df[feature_cols]
    y_train = train_file_df[DIST_COL].astype(float)

    X_val = val_file_df[feature_cols]
    y_val = val_file_df[DIST_COL].astype(float)

    X_test = test_file_df[feature_cols]
    y_test = test_file_df[DIST_COL].astype(float)

    # --------------------------------------------------------
    # Train all ML models
    # --------------------------------------------------------

    models = get_models()
    results = []

    best_model = None
    best_name = None
    best_val_mae = 1e9

    all_predictions = test_file_df.copy()

    for name, model in models.items():

        print(f"\nTraining {name} distance model...")

        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        val_pred = np.clip(val_pred, 0, None)
        test_pred = np.clip(test_pred, 0, None)

        val_mae, val_rmse, val_r2 = evaluate(
            y_val,
            val_pred,
            f"{name} VALIDATION"
        )

        test_mae, test_rmse, test_r2 = evaluate(
            y_test,
            test_pred,
            f"{name} FINAL TEST FILE-LEVEL"
        )

        results.append({
            "model": name,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2
        })

        all_predictions[f"{name}_distance_pred_m"] = test_pred
        all_predictions[f"{name}_abs_error_m"] = np.abs(y_test.values - test_pred)

        # Save individual model
        model_path = os.path.join(MODEL_DIR, f"filelevel_rssi_{name}.pkl")

        joblib.dump({
            "model_name": name,
            "model": model,
            "features": feature_cols,
            "window_rssi_features": rssi_features,
            "distance_col": DIST_COL,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
            "val_r2": val_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2
        }, model_path)

        print(f"Saved model: {model_path}")

        # Save and show true vs predicted graph
        save_true_vs_pred(
            y_test.values,
            test_pred,
            os.path.join(GRAPH_DIR, f"{name}_true_vs_pred.png"),
            f"{name}: True vs Predicted Distance"
        )

        # Save and show error histogram
        save_error_hist(
            np.abs(y_test.values - test_pred),
            os.path.join(GRAPH_DIR, f"{name}_error_histogram.png"),
            f"{name}: Distance Error Histogram"
        )

        # Save and show confusion matrix
        save_distance_confusion_matrix(
            y_test.values,
            test_pred,
            os.path.join(GRAPH_DIR, f"{name}_distance_confusion_matrix.png"),
            f"{name}: Distance Confusion Matrix"
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model = model
            best_name = name

    # --------------------------------------------------------
    # Save final comparison table
    # --------------------------------------------------------

    results_df = pd.DataFrame(results).sort_values("test_mae")

    results_path = os.path.join(
        OUT_DIR,
        "final_test_filelevel_mae_rmse_r2_comparison.csv"
    )

    results_df.to_csv(results_path, index=False)

    print("\n================ FINAL TEST FILE-LEVEL COMPARISON ================")
    print(results_df)

    print("\nSaved final test comparison table:")
    print(results_path)

    # --------------------------------------------------------
    # Save all model predictions
    # --------------------------------------------------------

    pred_path = os.path.join(
        OUT_DIR,
        "final_test_filelevel_all_model_predictions.csv"
    )

    all_predictions.to_csv(pred_path, index=False)

    print("\nSaved all final test predictions:")
    print(pred_path)

    # --------------------------------------------------------
    # Save and show model comparison graph
    # --------------------------------------------------------

    save_model_comparison_plot(
        results_df,
        os.path.join(GRAPH_DIR, "final_test_all_models_mae_comparison.png")
    )

    print("\nSaved all graphs in:")
    print(GRAPH_DIR)

    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    best_model_path = os.path.join(
        MODEL_DIR,
        "best_filelevel_rssi_distance_model.pkl"
    )

    joblib.dump({
        "best_model_name": best_name,
        "model": best_model,
        "features": feature_cols,
        "window_rssi_features": rssi_features,
        "distance_col": DIST_COL,
        "comparison_results": results_df
    }, best_model_path)

    print(f"\nBest model selected by validation MAE: {best_name}")
    print("Saved best model:")
    print(best_model_path)

    # --------------------------------------------------------
    # Feature importance for best tree-based model
    # --------------------------------------------------------

    final_estimator = best_model.named_steps["model"]

    if hasattr(final_estimator, "feature_importances_"):
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": final_estimator.feature_importances_,
        }).sort_values("importance", ascending=False)

        imp_path = os.path.join(
            OUT_DIR,
            "best_model_feature_importance.csv"
        )

        imp_df.to_csv(imp_path, index=False)

        print("\nTop 20 important distance features:")
        print(imp_df.head(20))

        print("\nSaved feature importance:")
        print(imp_path)

    print("\nRSSI file-level final test model comparison completed successfully.")


if __name__ == "__main__":
    main()