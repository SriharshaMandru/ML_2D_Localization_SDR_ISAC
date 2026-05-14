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
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = "data/processed/train_dataset.csv"
VAL_FILE   = "data/processed/val_dataset.csv"
TEST_FILE  = "data/processed/test_dataset.csv"

OUT_DIR = "results/xgboost_rssi_distance"
GRAPH_DIR = os.path.join(OUT_DIR, "graphs")
MODEL_DIR = "models_saved"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DIST_COL = "distance_m"
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


def select_rssi_distance_features(df):
    include_keywords = [
        "frequency_mhz",
        "rssi",
        "power",
        "amp",
    ]

    exclude_keywords = [
        "angle_deg",
        "distance_m",
        "x_true",
        "y_true",
        "x_pred",
        "y_pred",
        "phase",
        "cov",
        "r12",
        "r11",
        "r22",
        "aoa",
        "corr",
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

    if len(features) == 0:
        raise ValueError("No RSSI/distance features found. Check dataset columns.")

    return sorted(list(set(features)))


def evaluate_distance(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n================ {name} RESULTS ================")
    print(f"MAE  : {mae:.4f} m")
    print(f"RMSE : {rmse:.4f} m")
    print(f"R²   : {r2:.4f}")

    return mae, rmse, r2


def file_level_distance(pred_df):
    if "rx_file_path" in pred_df.columns:
        group_cols = ["rx_file_path"]
    elif "rx_file_name" in pred_df.columns:
        group_cols = ["rx_file_name"]
    else:
        group_cols = [
            c for c in ["frequency_mhz", "distance_m", "angle_deg", "rep"]
            if c in pred_df.columns
        ]

    rows = []

    for key, g in pred_df.groupby(group_cols):
        true_dist = g[DIST_COL].iloc[0]
        pred_dist = np.median(g["distance_pred_m"])

        rows.append({
            "group": str(key),
            "true_distance_m": true_dist,
            "pred_distance_m": pred_dist,
            "abs_error_m": abs(pred_dist - true_dist),
            "num_windows": len(g),
        })

    return pd.DataFrame(rows)


# ============================================================
# IEEE PLOTS
# ============================================================

def save_true_vs_pred_plot(y_true, y_pred, path, title):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    plt.figure(figsize=(6, 6))

    plt.scatter(
        y_true,
        y_pred,
        alpha=0.65,
        edgecolors="black",
        linewidths=0.4
    )

    min_v = min(np.min(y_true), np.min(y_pred))
    max_v = max(np.max(y_true), np.max(y_pred))

    plt.plot(
        [min_v, max_v],
        [min_v, max_v],
        "k--",
        linewidth=1.5,
        label="Ideal Prediction"
    )

    plt.xlabel("True Distance (m)")
    plt.ylabel("Predicted Distance (m)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_error_histogram(errors, path, title):
    plt.figure(figsize=(7, 5))

    plt.hist(errors, bins=30, edgecolor="black")

    plt.xlabel("Absolute Distance Error (m)")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_distance_confusion_matrix(y_true, y_pred, path, title):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    true_classes = np.sort(np.unique(y_true))

    y_pred_class = []

    for p in y_pred:
        nearest = true_classes[np.argmin(np.abs(true_classes - p))]
        y_pred_class.append(nearest)

    y_pred_class = np.array(y_pred_class)

    cm = confusion_matrix(y_true, y_pred_class, labels=true_classes)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted Distance Class (m)")
    plt.ylabel("True Distance Class (m)")

    plt.xticks(np.arange(len(true_classes)), true_classes)
    plt.yticks(np.arange(len(true_classes)), true_classes)

    plt.colorbar(label="Number of Samples")

    for i in range(len(true_classes)):
        for j in range(len(true_classes)):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=10
            )

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_distance_mae_bar(pred_df, path):
    mae_df = pred_df.groupby(DIST_COL)["distance_abs_error_m"].mean()

    plt.figure(figsize=(7, 4))

    mae_df.plot(kind="bar", edgecolor="black")

    plt.xlabel("True Distance (m)")
    plt.ylabel("Mean Absolute Error (m)")
    plt.title("Distance-wise MAE - XGBoost RSSI")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n===================================================")
    print(" XGBOOST RSSI-BASED DISTANCE ESTIMATION")
    print(" WITH IEEE STANDARD GRAPHS")
    print("===================================================\n")

    train_df = read_csv(TRAIN_FILE)
    val_df   = read_csv(VAL_FILE)
    test_df  = read_csv(TEST_FILE)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if DIST_COL not in df.columns:
            raise ValueError(f"{DIST_COL} missing in {name} dataset.")

    feature_cols = select_rssi_distance_features(train_df)

    print("\nSelected RSSI-distance features:")
    for c in feature_cols:
        print(" -", c)

    selected_features_path = os.path.join(
        OUT_DIR,
        "selected_distance_features.txt"
    )

    with open(selected_features_path, "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    print("\nSaved selected features:")
    print(selected_features_path)

    X_train = train_df[feature_cols]
    y_train = train_df[DIST_COL].astype(float)

    X_val = val_df[feature_cols]
    y_val = val_df[DIST_COL].astype(float)

    X_test = test_df[feature_cols]
    y_test = test_df[DIST_COL].astype(float)

    # ========================================================
    # XGBOOST MODEL
    # ========================================================

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=2,
            reg_lambda=2.0,
            reg_alpha=0.1,
        )),
    ])

    print("\nTraining XGBoost RSSI distance model...")
    model.fit(X_train, y_train)

    # ========================================================
    # VALIDATION
    # ========================================================

    val_pred = model.predict(X_val)
    val_pred = np.clip(val_pred, 0, None)

    evaluate_distance(y_val, val_pred, "VALIDATION WINDOW")

    # ========================================================
    # TEST WINDOW LEVEL
    # ========================================================

    test_pred = model.predict(X_test)
    test_pred = np.clip(test_pred, 0, None)

    test_mae, test_rmse, test_r2 = evaluate_distance(
        y_test,
        test_pred,
        "TEST WINDOW"
    )

    # ========================================================
    # SAVE WINDOW PREDICTIONS
    # ========================================================

    pred_df = test_df.copy()

    pred_df["distance_pred_m"] = test_pred
    pred_df["distance_abs_error_m"] = np.abs(
        pred_df[DIST_COL] - pred_df["distance_pred_m"]
    )

    pred_path = os.path.join(
        OUT_DIR,
        "distance_window_predictions.csv"
    )

    pred_df.to_csv(pred_path, index=False)

    print("\nSaved window predictions:")
    print(pred_path)

    # ========================================================
    # FILE LEVEL DISTANCE
    # ========================================================

    file_df = file_level_distance(pred_df)

    file_mae = mean_absolute_error(
        file_df["true_distance_m"],
        file_df["pred_distance_m"]
    )

    file_rmse = np.sqrt(mean_squared_error(
        file_df["true_distance_m"],
        file_df["pred_distance_m"]
    ))

    file_r2 = r2_score(
        file_df["true_distance_m"],
        file_df["pred_distance_m"]
    )

    print("\n================ FILE LEVEL DISTANCE RESULTS ================")
    print(f"MAE  : {file_mae:.4f} m")
    print(f"RMSE : {file_rmse:.4f} m")
    print(f"R²   : {file_r2:.4f}")

    file_path = os.path.join(
        OUT_DIR,
        "distance_file_level_predictions.csv"
    )

    file_df.to_csv(file_path, index=False)

    print("\nSaved file-level predictions:")
    print(file_path)

    # ========================================================
    # SAVE IEEE GRAPHS
    # ========================================================

    true_pred_window_path = os.path.join(
        GRAPH_DIR,
        "distance_true_vs_pred_window.png"
    )

    true_pred_file_path = os.path.join(
        GRAPH_DIR,
        "distance_true_vs_pred_file_level.png"
    )

    error_hist_window_path = os.path.join(
        GRAPH_DIR,
        "distance_error_histogram_window.png"
    )

    error_hist_file_path = os.path.join(
        GRAPH_DIR,
        "distance_error_histogram_file_level.png"
    )

    confusion_window_path = os.path.join(
        GRAPH_DIR,
        "distance_confusion_matrix_window.png"
    )

    confusion_file_path = os.path.join(
        GRAPH_DIR,
        "distance_confusion_matrix_file_level.png"
    )

    mae_bar_path = os.path.join(
        GRAPH_DIR,
        "distance_wise_mae.png"
    )

    save_true_vs_pred_plot(
        y_test.values,
        test_pred,
        true_pred_window_path,
        "True vs Predicted Distance - Window Level"
    )

    save_true_vs_pred_plot(
        file_df["true_distance_m"].values,
        file_df["pred_distance_m"].values,
        true_pred_file_path,
        "True vs Predicted Distance - File Level"
    )

    save_error_histogram(
        pred_df["distance_abs_error_m"].values,
        error_hist_window_path,
        "Distance Error Histogram - Window Level"
    )

    save_error_histogram(
        file_df["abs_error_m"].values,
        error_hist_file_path,
        "Distance Error Histogram - File Level"
    )

    save_distance_confusion_matrix(
        y_test.values,
        test_pred,
        confusion_window_path,
        "RSSI Distance Confusion Matrix - Window Level"
    )

    save_distance_confusion_matrix(
        file_df["true_distance_m"].values,
        file_df["pred_distance_m"].values,
        confusion_file_path,
        "RSSI Distance Confusion Matrix - File Level"
    )

    save_distance_mae_bar(
        pred_df,
        mae_bar_path
    )

    print("\nSaved IEEE graphs:")
    print(true_pred_window_path)
    print(true_pred_file_path)
    print(error_hist_window_path)
    print(error_hist_file_path)
    print(confusion_window_path)
    print(confusion_file_path)
    print(mae_bar_path)

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    xgb_model = model.named_steps["xgb"]

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": xgb_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    imp_path = os.path.join(
        OUT_DIR,
        "distance_feature_importance.csv"
    )

    imp_df.to_csv(imp_path, index=False)

    print("\nTop distance features:")
    print(imp_df.head(20))

    print("\nSaved feature importance:")
    print(imp_path)

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "distance_col": DIST_COL,
        "test_window_mae": test_mae,
        "test_window_rmse": test_rmse,
        "test_window_r2": test_r2,
        "file_level_mae": file_mae,
        "file_level_rmse": file_rmse,
        "file_level_r2": file_r2,
    }

    model_path = os.path.join(
        MODEL_DIR,
        "xgboost_rssi_distance.pkl"
    )

    joblib.dump(model_bundle, model_path)

    print("\nSaved distance model:")
    print(model_path)

    print("\nXGBoost RSSI distance training completed successfully.")


if __name__ == "__main__":
    main()