#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
TRAIN_FILE = "data/processed/train_dataset.csv"
VAL_FILE   = "data/processed/val_dataset.csv"
TEST_FILE  = "data/processed/test_dataset.csv"

OUT_DIR = "results/xgboost_2d_localization"
GRAPH_DIR = os.path.join(OUT_DIR, "graphs")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

ANGLE_COL = "angle_deg"
DIST_COL  = "distance_m"

RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {path}: {df.shape}")
    return df


def angle_distance_to_xy(angle_deg, distance_m):
    theta = np.deg2rad(angle_deg)
    x = distance_m * np.sin(theta)
    y = distance_m * np.cos(theta)
    return x, y


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def find_feature_columns(df):
    remove_cols = [
        ANGLE_COL, DIST_COL,
        "x_true", "y_true", "x_pred", "y_pred",
        "split", "label", "target",
        "rx_file_path", "rx_file_name",
        "tx_file_path", "tx_file_name",
        "file_path", "file_name"
    ]

    feature_cols = []

    for col in df.columns:
        if col in remove_cols:
            continue

        if df[col].dtype == "object":
            continue

        if df[col].isna().all():
            continue

        feature_cols.append(col)

    return feature_cols


def clean_features(train_df, val_df, test_df, feature_cols):
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_val   = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test  = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return X_train, X_val, X_test


def get_file_id(df):
    if "rx_file_path" in df.columns:
        return df["rx_file_path"].astype(str)

    if "rx_file_name" in df.columns:
        base = df["rx_file_name"].astype(str)
    else:
        base = pd.Series(np.arange(len(df)), index=df.index).astype(str)

    parts = []

    for col in ["freq_mhz", "distance_m", "angle_deg", "pair", "rep"]:
        if col in df.columns:
            parts.append(df[col].astype(str))

    if len(parts) > 0:
        file_id = parts[0]
        for p in parts[1:]:
            file_id = file_id + "_" + p
        return file_id

    return base


def majority_vote(values):
    return pd.Series(values).value_counts().idxmax()


# ============================================================
# MAIN
# ============================================================
def main():

    print("\n===================================================")
    print(" XGBOOST HYBRID AoA + RSSI 2D LOCALIZATION")
    print("===================================================\n")

    train_df = load_csv(TRAIN_FILE)
    val_df   = load_csv(VAL_FILE)
    test_df  = load_csv(TEST_FILE)

    if ANGLE_COL not in train_df.columns:
        raise ValueError(f"Missing angle column: {ANGLE_COL}")

    if DIST_COL not in train_df.columns:
        raise ValueError(f"Missing distance column: {DIST_COL}")

    # ------------------------------------------------------------
    # FEATURE SELECTION
    # ------------------------------------------------------------
    feature_cols = find_feature_columns(train_df)

    print(f"\nTotal selected features: {len(feature_cols)}")

    with open(os.path.join(OUT_DIR, "used_features.txt"), "w") as f:
        for col in feature_cols:
            f.write(col + "\n")

    X_train, X_val, X_test = clean_features(train_df, val_df, test_df, feature_cols)

    # Combine train + validation for final model training
    X_train_full = pd.concat([X_train, X_val], axis=0)
    train_full_df = pd.concat([train_df, val_df], axis=0)

    y_angle_train = train_full_df[ANGLE_COL].values
    y_dist_train  = train_full_df[DIST_COL].values

    y_angle_test = test_df[ANGLE_COL].values
    y_dist_test  = test_df[DIST_COL].values

    # ------------------------------------------------------------
    # ENCODE ANGLES FOR XGBOOST CLASSIFIER
    # ------------------------------------------------------------
    label_encoder = LabelEncoder()
    y_angle_train_enc = label_encoder.fit_transform(y_angle_train)

    # ------------------------------------------------------------
    # TRAIN XGBOOST AoA CLASSIFIER
    # ------------------------------------------------------------
    print("\nTraining XGBoost AoA classifier...")

    aoa_model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    aoa_model.fit(X_train_full, y_angle_train_enc)

    angle_pred_enc = aoa_model.predict(X_test)
    angle_pred = label_encoder.inverse_transform(angle_pred_enc)

    angle_accuracy = accuracy_score(y_angle_test, angle_pred) * 100
    angle_mae = mean_absolute_error(y_angle_test, angle_pred)
    angle_rmse = rmse(y_angle_test, angle_pred)

    print("\nAoA Results:")
    print(f"Accuracy : {angle_accuracy:.2f} %")
    print(f"MAE      : {angle_mae:.4f} deg")
    print(f"RMSE     : {angle_rmse:.4f} deg")

    # ------------------------------------------------------------
    # TRAIN XGBOOST RSSI / DISTANCE REGRESSOR
    # ------------------------------------------------------------
    print("\nTraining XGBoost distance regressor...")

    dist_model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    dist_model.fit(X_train_full, y_dist_train)

    dist_pred = dist_model.predict(X_test)
    dist_pred = np.maximum(dist_pred, 0.01)

    dist_mae = mean_absolute_error(y_dist_test, dist_pred)
    dist_rmse = rmse(y_dist_test, dist_pred)
    dist_r2 = r2_score(y_dist_test, dist_pred)

    print("\nDistance Results:")
    print(f"MAE  : {dist_mae:.4f} m")
    print(f"RMSE : {dist_rmse:.4f} m")
    print(f"R2   : {dist_r2:.4f}")

    # ------------------------------------------------------------
    # CONVERT AoA + DISTANCE TO 2D COORDINATES
    # ------------------------------------------------------------
    x_true, y_true = angle_distance_to_xy(y_angle_test, y_dist_test)
    x_pred, y_pred = angle_distance_to_xy(angle_pred, dist_pred)

    pos_error = np.sqrt((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2)

    pos_mae = np.mean(pos_error)
    pos_rmse = np.sqrt(np.mean(pos_error ** 2))
    pos_median = np.median(pos_error)
    pos_max = np.max(pos_error)
    cep50 = np.percentile(pos_error, 50)
    cep90 = np.percentile(pos_error, 90)

    print("\n2D Localization Results:")
    print(f"Mean Position Error   : {pos_mae:.4f} m")
    print(f"RMSE Position Error   : {pos_rmse:.4f} m")
    print(f"Median Position Error : {pos_median:.4f} m")
    print(f"Max Position Error    : {pos_max:.4f} m")
    print(f"CEP50                 : {cep50:.4f} m")
    print(f"CEP90                 : {cep90:.4f} m")

    # ------------------------------------------------------------
    # SAVE WINDOW-LEVEL PREDICTIONS
    # ------------------------------------------------------------
    results_df = test_df.copy()

    results_df["true_angle_deg"] = y_angle_test
    results_df["pred_angle_deg"] = angle_pred
    results_df["angle_error_deg"] = np.abs(results_df["true_angle_deg"] - results_df["pred_angle_deg"])

    results_df["true_distance_m"] = y_dist_test
    results_df["pred_distance_m"] = dist_pred
    results_df["distance_error_m"] = np.abs(results_df["true_distance_m"] - results_df["pred_distance_m"])

    results_df["x_true"] = x_true
    results_df["y_true"] = y_true
    results_df["x_pred"] = x_pred
    results_df["y_pred"] = y_pred
    results_df["position_error_m"] = pos_error

    results_df["file_id"] = get_file_id(results_df)

    pred_file = os.path.join(OUT_DIR, "xgboost_2d_window_predictions.csv")
    results_df.to_csv(pred_file, index=False)

    print(f"\nSaved window predictions:\n{pred_file}")

    # ------------------------------------------------------------
    # FILE-LEVEL AGGREGATION
    # ------------------------------------------------------------
    file_rows = []

    for file_id, g in results_df.groupby("file_id"):

        true_angle = majority_vote(g["true_angle_deg"])
        pred_angle = majority_vote(g["pred_angle_deg"])

        true_dist = g["true_distance_m"].median()
        pred_dist = g["pred_distance_m"].median()

        fx_true, fy_true = angle_distance_to_xy(true_angle, true_dist)
        fx_pred, fy_pred = angle_distance_to_xy(pred_angle, pred_dist)

        f_pos_error = np.sqrt((fx_true - fx_pred) ** 2 + (fy_true - fy_pred) ** 2)

        row = {
            "file_id": file_id,
            "true_angle_deg": true_angle,
            "pred_angle_deg": pred_angle,
            "angle_error_deg": abs(true_angle - pred_angle),
            "true_distance_m": true_dist,
            "pred_distance_m": pred_dist,
            "distance_error_m": abs(true_dist - pred_dist),
            "x_true": fx_true,
            "y_true": fy_true,
            "x_pred": fx_pred,
            "y_pred": fy_pred,
            "position_error_m": f_pos_error,
            "num_windows": len(g)
        }

        for col in ["freq_mhz", "pair", "rep"]:
            if col in g.columns:
                row[col] = g[col].iloc[0]

        file_rows.append(row)

    file_df = pd.DataFrame(file_rows)

    file_angle_acc = accuracy_score(file_df["true_angle_deg"], file_df["pred_angle_deg"]) * 100
    file_angle_mae = mean_absolute_error(file_df["true_angle_deg"], file_df["pred_angle_deg"])

    file_dist_mae = mean_absolute_error(file_df["true_distance_m"], file_df["pred_distance_m"])
    file_dist_rmse = rmse(file_df["true_distance_m"], file_df["pred_distance_m"])

    file_pos_mean = file_df["position_error_m"].mean()
    file_pos_rmse = np.sqrt(np.mean(file_df["position_error_m"] ** 2))
    file_pos_median = file_df["position_error_m"].median()
    file_cep90 = np.percentile(file_df["position_error_m"], 90)

    file_pred_file = os.path.join(OUT_DIR, "xgboost_2d_file_predictions.csv")
    file_df.to_csv(file_pred_file, index=False)

    print(f"\nSaved file-level predictions:\n{file_pred_file}")

    print("\nFile-Level Final Results:")
    print(f"File AoA Accuracy        : {file_angle_acc:.2f} %")
    print(f"File AoA MAE             : {file_angle_mae:.4f} deg")
    print(f"File Distance MAE        : {file_dist_mae:.4f} m")
    print(f"File Distance RMSE       : {file_dist_rmse:.4f} m")
    print(f"File Mean Position Error : {file_pos_mean:.4f} m")
    print(f"File RMSE Position Error : {file_pos_rmse:.4f} m")
    print(f"File Median Error        : {file_pos_median:.4f} m")
    print(f"File CEP90               : {file_cep90:.4f} m")

    # ------------------------------------------------------------
    # SAVE SUMMARY TABLE
    # ------------------------------------------------------------
    summary = pd.DataFrame([
        {
            "level": "window",
            "aoa_accuracy_percent": angle_accuracy,
            "aoa_mae_deg": angle_mae,
            "aoa_rmse_deg": angle_rmse,
            "distance_mae_m": dist_mae,
            "distance_rmse_m": dist_rmse,
            "distance_r2": dist_r2,
            "mean_position_error_m": pos_mae,
            "rmse_position_error_m": pos_rmse,
            "median_position_error_m": pos_median,
            "cep50_m": cep50,
            "cep90_m": cep90,
            "num_samples": len(results_df)
        },
        {
            "level": "file",
            "aoa_accuracy_percent": file_angle_acc,
            "aoa_mae_deg": file_angle_mae,
            "aoa_rmse_deg": np.nan,
            "distance_mae_m": file_dist_mae,
            "distance_rmse_m": file_dist_rmse,
            "distance_r2": np.nan,
            "mean_position_error_m": file_pos_mean,
            "rmse_position_error_m": file_pos_rmse,
            "median_position_error_m": file_pos_median,
            "cep50_m": file_pos_median,
            "cep90_m": file_cep90,
            "num_samples": len(file_df)
        }
    ])

    summary_file = os.path.join(OUT_DIR, "xgboost_2d_localization_summary.csv")
    summary.to_csv(summary_file, index=False)

    print(f"\nSaved summary:\n{summary_file}")

    # ------------------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------------------
    report = classification_report(
        y_angle_test,
        angle_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(OUT_DIR, "xgboost_aoa_classification_report.csv"))

    cm = confusion_matrix(y_angle_test, angle_pred, labels=label_encoder.classes_)
    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )
    cm_df.to_csv(os.path.join(OUT_DIR, "xgboost_aoa_confusion_matrix.csv"))

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------

    # 1. 2D localization map
    plt.figure(figsize=(7, 7))
    plt.scatter(file_df["x_true"], file_df["y_true"], marker="o", label="True Position")
    plt.scatter(file_df["x_pred"], file_df["y_pred"], marker="x", label="Predicted Position")

    for _, row in file_df.iterrows():
        plt.plot(
            [row["x_true"], row["x_pred"]],
            [row["y_true"], row["y_pred"]],
            linewidth=0.8,
            alpha=0.6
        )

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("XGBoost Hybrid AoA + RSSI 2D Localization")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "xgboost_2d_localization_map.png"), dpi=300)
    plt.close()

    # 2. Position error CDF
    sorted_err = np.sort(file_df["position_error_m"].values)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)

    plt.figure(figsize=(7, 5))
    plt.plot(sorted_err, cdf, marker="o")
    plt.xlabel("Position Error (m)")
    plt.ylabel("CDF")
    plt.title("CDF of 2D Localization Error")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "position_error_cdf.png"), dpi=300)
    plt.close()

    # 3. True vs predicted distance
    plt.figure(figsize=(6, 6))
    plt.scatter(file_df["true_distance_m"], file_df["pred_distance_m"])
    min_d = min(file_df["true_distance_m"].min(), file_df["pred_distance_m"].min())
    max_d = max(file_df["true_distance_m"].max(), file_df["pred_distance_m"].max())
    plt.plot([min_d, max_d], [min_d, max_d], linestyle="--")
    plt.xlabel("True Distance (m)")
    plt.ylabel("Predicted Distance (m)")
    plt.title("XGBoost RSSI Distance Estimation")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "true_vs_predicted_distance.png"), dpi=300)
    plt.close()

    # 4. True vs predicted AoA
    plt.figure(figsize=(6, 6))
    plt.scatter(file_df["true_angle_deg"], file_df["pred_angle_deg"])
    min_a = min(file_df["true_angle_deg"].min(), file_df["pred_angle_deg"].min())
    max_a = max(file_df["true_angle_deg"].max(), file_df["pred_angle_deg"].max())
    plt.plot([min_a, max_a], [min_a, max_a], linestyle="--")
    plt.xlabel("True AoA (deg)")
    plt.ylabel("Predicted AoA (deg)")
    plt.title("XGBoost AoA Estimation")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "true_vs_predicted_aoa.png"), dpi=300)
    plt.close()

    # 5. Position error bar graph
    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(len(file_df)), file_df["position_error_m"])
    plt.xlabel("Test File Index")
    plt.ylabel("Position Error (m)")
    plt.title("File-Level 2D Localization Error")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "file_position_error_bar.png"), dpi=300)
    plt.close()

    print(f"\nSaved graphs in:\n{GRAPH_DIR}")
    print("\nXGBoost 2D localization completed successfully.")


if __name__ == "__main__":
    main()