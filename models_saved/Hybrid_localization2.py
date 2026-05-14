#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================
TRAIN_FILE = "data/processed/train_dataset.csv"
VAL_FILE   = "data/processed/val_dataset.csv"
TEST_FILE  = "data/processed/test_dataset.csv"

OUT_DIR = "results/hybrid_xgboost_aoa_extratrees_rssi_2d"
GRAPH_DIR = os.path.join(OUT_DIR, "graphs")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

ANGLE_COL = "angle_deg"
DIST_COL  = "distance_m"

RANDOM_STATE = 42


# ============================================================
# BASIC FUNCTIONS
# ============================================================
def load_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {path}: {df.shape}")
    return df


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def angle_distance_to_xy(angle_deg, distance_m):
    theta = np.deg2rad(angle_deg)
    x = distance_m * np.sin(theta)
    y = distance_m * np.cos(theta)
    return x, y


def majority_vote(values):
    return pd.Series(values).value_counts().idxmax()


def get_file_id(df):
    parts = []

    for col in ["frequency_mhz", "freq_mhz", "distance_m", "angle_deg", "rep"]:
        if col in df.columns:
            parts.append(df[col].astype(str))

    if len(parts) > 0:
        file_id = parts[0]
        for p in parts[1:]:
            file_id = file_id + "_" + p
        return file_id

    if "rx_file_path" in df.columns:
        return df["rx_file_path"].astype(str)

    if "rx_file_name" in df.columns:
        return df["rx_file_name"].astype(str)

    return pd.Series(np.arange(len(df)), index=df.index).astype(str)


def select_numeric_features(df):
    remove_cols = [
        ANGLE_COL, DIST_COL,
        "x_true", "y_true", "x_pred", "y_pred",
        "true_angle_deg", "pred_angle_deg",
        "true_distance_m", "pred_distance_m",
        "angle_error_deg", "distance_error_m",
        "position_error_m",
        "split", "label", "target",
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


def clean_X(df, feature_cols):
    return df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)


# ============================================================
# MAIN
# ============================================================
def main():

    print("\n===================================================")
    print(" HYBRID 2D LOCALIZATION")
    print(" XGBoost AoA + ExtraTrees RSSI Distance")
    print("===================================================\n")

    train_df = load_csv(TRAIN_FILE)
    val_df   = load_csv(VAL_FILE)
    test_df  = load_csv(TEST_FILE)

    train_full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    # ------------------------------------------------------------
    # FEATURE SELECTION
    # ------------------------------------------------------------
    feature_cols = select_numeric_features(train_full_df)

    print(f"\nTotal selected features: {len(feature_cols)}")

    with open(os.path.join(OUT_DIR, "used_features.txt"), "w") as f:
        for c in feature_cols:
            f.write(c + "\n")

    X_train = clean_X(train_full_df, feature_cols)
    X_test  = clean_X(test_df, feature_cols)

    y_angle_train = train_full_df[ANGLE_COL].values
    y_angle_test  = test_df[ANGLE_COL].values

    y_dist_train = train_full_df[DIST_COL].values
    y_dist_test  = test_df[DIST_COL].values

    # ------------------------------------------------------------
    # XGBOOST AoA CLASSIFIER
    # ------------------------------------------------------------
    print("\nTraining XGBoost AoA classifier...")

    le = LabelEncoder()
    y_angle_train_enc = le.fit_transform(y_angle_train)

    aoa_model = XGBClassifier(
        n_estimators=450,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    aoa_model.fit(X_train, y_angle_train_enc)

    pred_angle_enc = aoa_model.predict(X_test)
    pred_angle = le.inverse_transform(pred_angle_enc)

    window_aoa_acc = accuracy_score(y_angle_test, pred_angle) * 100
    window_aoa_mae = mean_absolute_error(y_angle_test, pred_angle)
    window_aoa_rmse = rmse(y_angle_test, pred_angle)

    print("\nWindow-Level AoA Results:")
    print(f"Accuracy : {window_aoa_acc:.2f} %")
    print(f"MAE      : {window_aoa_mae:.4f} deg")
    print(f"RMSE     : {window_aoa_rmse:.4f} deg")

    # ------------------------------------------------------------
    # EXTRATREES RSSI/DISTANCE REGRESSOR
    # ------------------------------------------------------------
    print("\nTraining ExtraTrees RSSI distance regressor...")

    dist_model = ExtraTreesRegressor(
        n_estimators=600,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    dist_model.fit(X_train, y_dist_train)

    pred_dist = dist_model.predict(X_test)
    pred_dist = np.maximum(pred_dist, 0.01)

    window_dist_mae = mean_absolute_error(y_dist_test, pred_dist)
    window_dist_rmse = rmse(y_dist_test, pred_dist)
    window_dist_r2 = r2_score(y_dist_test, pred_dist)

    print("\nWindow-Level Distance Results:")
    print(f"MAE  : {window_dist_mae:.4f} m")
    print(f"RMSE : {window_dist_rmse:.4f} m")
    print(f"R2   : {window_dist_r2:.4f}")

    # ------------------------------------------------------------
    # WINDOW-LEVEL 2D LOCALIZATION
    # ------------------------------------------------------------
    x_true, y_true = angle_distance_to_xy(y_angle_test, y_dist_test)
    x_pred, y_pred = angle_distance_to_xy(pred_angle, pred_dist)

    pos_error = np.sqrt((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2)

    window_pos_mean = np.mean(pos_error)
    window_pos_rmse = np.sqrt(np.mean(pos_error ** 2))
    window_pos_median = np.median(pos_error)
    window_cep50 = np.percentile(pos_error, 50)
    window_cep90 = np.percentile(pos_error, 90)

    print("\nWindow-Level 2D Localization Results:")
    print(f"Mean Position Error   : {window_pos_mean:.4f} m")
    print(f"RMSE Position Error   : {window_pos_rmse:.4f} m")
    print(f"Median Position Error : {window_pos_median:.4f} m")
    print(f"CEP50                 : {window_cep50:.4f} m")
    print(f"CEP90                 : {window_cep90:.4f} m")

    # ------------------------------------------------------------
    # SAVE WINDOW RESULTS
    # ------------------------------------------------------------
    results_df = test_df.copy()

    results_df["true_angle_deg"] = y_angle_test
    results_df["pred_angle_deg"] = pred_angle
    results_df["angle_error_deg"] = np.abs(y_angle_test - pred_angle)

    results_df["true_distance_m"] = y_dist_test
    results_df["pred_distance_m"] = pred_dist
    results_df["distance_error_m"] = np.abs(y_dist_test - pred_dist)

    results_df["x_true"] = x_true
    results_df["y_true"] = y_true
    results_df["x_pred"] = x_pred
    results_df["y_pred"] = y_pred
    results_df["position_error_m"] = pos_error

    results_df["file_id"] = get_file_id(results_df)

    window_out = os.path.join(OUT_DIR, "hybrid_window_predictions.csv")
    results_df.to_csv(window_out, index=False)

    print(f"\nSaved window predictions:\n{window_out}")

    # ------------------------------------------------------------
    # FILE-LEVEL AGGREGATION
    # ------------------------------------------------------------
    file_rows = []

    for file_id, g in results_df.groupby("file_id"):

        true_angle = majority_vote(g["true_angle_deg"])
        pred_angle_file = majority_vote(g["pred_angle_deg"])

        true_dist = g["true_distance_m"].median()
        pred_dist_file = g["pred_distance_m"].median()

        fx_true, fy_true = angle_distance_to_xy(true_angle, true_dist)
        fx_pred, fy_pred = angle_distance_to_xy(pred_angle_file, pred_dist_file)

        f_error = np.sqrt((fx_true - fx_pred) ** 2 + (fy_true - fy_pred) ** 2)

        row = {
            "file_id": file_id,
            "true_angle_deg": true_angle,
            "pred_angle_deg": pred_angle_file,
            "angle_error_deg": abs(true_angle - pred_angle_file),
            "true_distance_m": true_dist,
            "pred_distance_m": pred_dist_file,
            "distance_error_m": abs(true_dist - pred_dist_file),
            "x_true": fx_true,
            "y_true": fy_true,
            "x_pred": fx_pred,
            "y_pred": fy_pred,
            "position_error_m": f_error,
            "num_windows": len(g)
        }

        for col in ["frequency_mhz", "freq_mhz", "rep"]:
            if col in g.columns:
                row[col] = g[col].iloc[0]

        file_rows.append(row)

    file_df = pd.DataFrame(file_rows)

    file_aoa_acc = accuracy_score(file_df["true_angle_deg"], file_df["pred_angle_deg"]) * 100
    file_aoa_mae = mean_absolute_error(file_df["true_angle_deg"], file_df["pred_angle_deg"])

    file_dist_mae = mean_absolute_error(file_df["true_distance_m"], file_df["pred_distance_m"])
    file_dist_rmse = rmse(file_df["true_distance_m"], file_df["pred_distance_m"])
    file_dist_r2 = r2_score(file_df["true_distance_m"], file_df["pred_distance_m"])

    file_pos_mean = file_df["position_error_m"].mean()
    file_pos_rmse = np.sqrt(np.mean(file_df["position_error_m"] ** 2))
    file_pos_median = file_df["position_error_m"].median()
    file_cep50 = np.percentile(file_df["position_error_m"], 50)
    file_cep90 = np.percentile(file_df["position_error_m"], 90)

    file_out = os.path.join(OUT_DIR, "hybrid_file_predictions.csv")
    file_df.to_csv(file_out, index=False)

    print(f"\nSaved file-level predictions:\n{file_out}")

    print("\nFile-Level Final Hybrid Results:")
    print(f"File AoA Accuracy        : {file_aoa_acc:.2f} %")
    print(f"File AoA MAE             : {file_aoa_mae:.4f} deg")
    print(f"File Distance MAE        : {file_dist_mae:.4f} m")
    print(f"File Distance RMSE       : {file_dist_rmse:.4f} m")
    print(f"File Distance R2         : {file_dist_r2:.4f}")
    print(f"File Mean Position Error : {file_pos_mean:.4f} m")
    print(f"File RMSE Position Error : {file_pos_rmse:.4f} m")
    print(f"File Median Error        : {file_pos_median:.4f} m")
    print(f"File CEP50               : {file_cep50:.4f} m")
    print(f"File CEP90               : {file_cep90:.4f} m")

    # ------------------------------------------------------------
    # SUMMARY CSV
    # ------------------------------------------------------------
    summary_df = pd.DataFrame([
        {
            "level": "window",
            "aoa_model": "XGBoostClassifier",
            "distance_model": "ExtraTreesRegressor",
            "aoa_accuracy_percent": window_aoa_acc,
            "aoa_mae_deg": window_aoa_mae,
            "aoa_rmse_deg": window_aoa_rmse,
            "distance_mae_m": window_dist_mae,
            "distance_rmse_m": window_dist_rmse,
            "distance_r2": window_dist_r2,
            "mean_position_error_m": window_pos_mean,
            "rmse_position_error_m": window_pos_rmse,
            "median_position_error_m": window_pos_median,
            "cep50_m": window_cep50,
            "cep90_m": window_cep90,
            "num_samples": len(results_df)
        },
        {
            "level": "file",
            "aoa_model": "XGBoostClassifier",
            "distance_model": "ExtraTreesRegressor",
            "aoa_accuracy_percent": file_aoa_acc,
            "aoa_mae_deg": file_aoa_mae,
            "aoa_rmse_deg": np.nan,
            "distance_mae_m": file_dist_mae,
            "distance_rmse_m": file_dist_rmse,
            "distance_r2": file_dist_r2,
            "mean_position_error_m": file_pos_mean,
            "rmse_position_error_m": file_pos_rmse,
            "median_position_error_m": file_pos_median,
            "cep50_m": file_cep50,
            "cep90_m": file_cep90,
            "num_samples": len(file_df)
        }
    ])

    summary_out = os.path.join(OUT_DIR, "hybrid_2d_localization_summary.csv")
    summary_df.to_csv(summary_out, index=False)

    print(f"\nSaved summary:\n{summary_out}")

    # ------------------------------------------------------------
    # AoA REPORTS
    # ------------------------------------------------------------
    report_df = pd.DataFrame(
        classification_report(
            y_angle_test,
            pred_angle,
            output_dict=True,
            zero_division=0
        )
    ).transpose()

    report_df.to_csv(os.path.join(OUT_DIR, "xgboost_aoa_classification_report.csv"))

    cm = confusion_matrix(y_angle_test, pred_angle, labels=le.classes_)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(os.path.join(OUT_DIR, "xgboost_aoa_confusion_matrix.csv"))

    # ------------------------------------------------------------
    # FEATURE IMPORTANCE FOR DISTANCE MODEL
    # ------------------------------------------------------------
    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": dist_model.feature_importances_
    }).sort_values("importance", ascending=False)

    imp_df.to_csv(os.path.join(OUT_DIR, "extratrees_distance_feature_importance.csv"), index=False)

    # ------------------------------------------------------------
    # GRAPHS
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

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Hybrid 2D Localization: XGBoost AoA + ExtraTrees RSSI")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "hybrid_2d_localization_map.png"), dpi=300)
    plt.close()

    # 2. Position error CDF
    sorted_err = np.sort(file_df["position_error_m"].values)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)

    plt.figure(figsize=(7, 5))
    plt.plot(sorted_err, cdf, marker="o")
    plt.xlabel("Position Error (m)")
    plt.ylabel("CDF")
    plt.title("CDF of Hybrid 2D Localization Error")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "hybrid_position_error_cdf.png"), dpi=300)
    plt.close()

    # 3. True vs predicted AoA
    plt.figure(figsize=(6, 6))
    plt.scatter(file_df["true_angle_deg"], file_df["pred_angle_deg"])
    min_a = min(file_df["true_angle_deg"].min(), file_df["pred_angle_deg"].min())
    max_a = max(file_df["true_angle_deg"].max(), file_df["pred_angle_deg"].max())
    plt.plot([min_a, max_a], [min_a, max_a], linestyle="--")
    plt.xlabel("True AoA (deg)")
    plt.ylabel("Predicted AoA (deg)")
    plt.title("File-Level AoA: XGBoost")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "file_true_vs_predicted_aoa.png"), dpi=300)
    plt.close()

    # 4. True vs predicted distance
    plt.figure(figsize=(6, 6))
    plt.scatter(file_df["true_distance_m"], file_df["pred_distance_m"])
    min_d = min(file_df["true_distance_m"].min(), file_df["pred_distance_m"].min())
    max_d = max(file_df["true_distance_m"].max(), file_df["pred_distance_m"].max())
    plt.plot([min_d, max_d], [min_d, max_d], linestyle="--")
    plt.xlabel("True Distance (m)")
    plt.ylabel("Predicted Distance (m)")
    plt.title("File-Level Distance: ExtraTrees")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "file_true_vs_predicted_distance.png"), dpi=300)
    plt.close()

    # 5. Bar plot position error
    plt.figure(figsize=(9, 5))
    plt.bar(np.arange(len(file_df)), file_df["position_error_m"])
    plt.xlabel("Test File Index")
    plt.ylabel("Position Error (m)")
    plt.title("File-Level Hybrid 2D Localization Error")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "file_position_error_bar.png"), dpi=300)
    plt.close()

    print(f"\nSaved graphs in:\n{GRAPH_DIR}")
    print("\nHybrid XGBoost AoA + ExtraTrees RSSI 2D localization completed successfully.")


if __name__ == "__main__":
    main()