#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ============================================================
# SAFE CPU SETTINGS
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# INPUT FILES
# ============================================================

TRAIN_FILE = "data/processed/train_dataset.csv"
VAL_FILE   = "data/processed/val_dataset.csv"
TEST_FILE  = "data/processed/test_dataset.csv"

AOA_FILE  = "results/aoa_fast_extratrees/aoa_file_level_predictions.csv"
DIST_FILE = "results/xgboost_rssi_distance/distance_file_level_predictions.csv"

OUT_DIR = "results/final_performance_comparison_with_hybrid"
GRAPH_DIR = os.path.join(OUT_DIR, "graphs")
MODEL_DIR = "models_saved"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ANGLE_COL = "angle_deg"
DIST_COL = "distance_m"
RANDOM_STATE = 42


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    print(f"Loaded {path}: {df.shape}", flush=True)
    return df


def angle_distance_to_xy(angle_deg, distance_m):
    theta = np.deg2rad(angle_deg)
    x = distance_m * np.sin(theta)
    y = distance_m * np.cos(theta)
    return x, y


def ensure_xy(df):
    df = df.copy()

    if "x_true" in df.columns and "y_true" in df.columns:
        df["x_true"] = pd.to_numeric(df["x_true"], errors="coerce")
        df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
        return df

    if ANGLE_COL not in df.columns or DIST_COL not in df.columns:
        raise ValueError("Need either x_true/y_true or angle_deg/distance_m")

    df["x_true"], df["y_true"] = angle_distance_to_xy(
        pd.to_numeric(df[ANGLE_COL], errors="coerce"),
        pd.to_numeric(df[DIST_COL], errors="coerce")
    )

    return df


def select_features(df):
    exclude_keywords = [
        "x_true", "y_true", "x_pred", "y_pred",
        "distance_m", "angle_deg",
        "split", "rep", "file", "path", "name",
        "tx_available"
    ]

    features = []

    for col in df.columns:
        low = col.lower()

        if any(k in low for k in exclude_keywords):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)

    if len(features) == 0:
        raise ValueError("No numeric features selected.")

    return sorted(features)


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


def file_level_from_window_predictions(pred_df, model_name):
    group_cols = get_group_cols(pred_df)

    rows = []

    for key, g in pred_df.groupby(group_cols):
        x_true = g["x_true"].iloc[0]
        y_true = g["y_true"].iloc[0]

        x_pred = np.median(g[f"{model_name}_x_pred"])
        y_pred = np.median(g[f"{model_name}_y_pred"])

        error_2d = np.sqrt((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2)

        rows.append({
            "group": str(key),
            "x_true": x_true,
            "y_true": y_true,
            "x_pred": x_pred,
            "y_pred": y_pred,
            "error_2d_m": error_2d,
        })

    return pd.DataFrame(rows)


def compute_metrics(df, model_name):
    y_true_xy = df[["x_true", "y_true"]].values
    y_pred_xy = df[["x_pred", "y_pred"]].values

    mse_x = mean_squared_error(df["x_true"], df["x_pred"])
    mse_y = mean_squared_error(df["y_true"], df["y_pred"])

    rmse_x = np.sqrt(mse_x)
    rmse_y = np.sqrt(mse_y)

    r2_x = r2_score(df["x_true"], df["x_pred"])
    r2_y = r2_score(df["y_true"], df["y_pred"])

    mse_2d = np.mean(df["error_2d_m"] ** 2)
    rmse_2d = np.sqrt(mse_2d)

    r2_2d = r2_score(y_true_xy, y_pred_xy)

    return {
        "model": model_name,
        "num_test_files": len(df),

        "mse_x": mse_x,
        "rmse_x_m": rmse_x,
        "r2_x": r2_x,

        "mse_y": mse_y,
        "rmse_y_m": rmse_y,
        "r2_y": r2_y,

        "mse_2d": mse_2d,
        "rmse_2d_m": rmse_2d,
        "r2_2d": r2_2d,

        "mean_2d_error_m": df["error_2d_m"].mean(),
        "median_2d_error_m": df["error_2d_m"].median(),
        "max_2d_error_m": df["error_2d_m"].max(),

        "within_0_5m_percent": np.mean(df["error_2d_m"] <= 0.5) * 100,
        "within_1m_percent": np.mean(df["error_2d_m"] <= 1.0) * 100,
        "within_2m_percent": np.mean(df["error_2d_m"] <= 2.0) * 100,
    }


# ============================================================
# MODELS
# ============================================================

def get_direct_models():
    models = {}

    models["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])

    models["ExtraTrees"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=1
        ))
    ])

    models["XGBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", MultiOutputRegressor(
            XGBRegressor(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.90,
                colsample_bytree=0.90,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=1,
                reg_lambda=2.0,
                reg_alpha=0.1
            )
        ))
    ])

    models["MLP"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=RANDOM_STATE
        ))
    ])

    models["SVM_LinearSVR"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", MultiOutputRegressor(
            LinearSVR(
                C=10,
                epsilon=0.05,
                max_iter=3000,
                random_state=RANDOM_STATE
            )
        ))
    ])

    return models


# ============================================================
# PROPOSED HYBRID FILE-LEVEL FUSION
# ============================================================

def run_filelevel_hybrid():
    aoa_df = read_csv(AOA_FILE)
    dist_df = read_csv(DIST_FILE)

    required_aoa = ["group", "true_angle_deg", "pred_angle_deg"]
    required_dist = ["group", "true_distance_m", "pred_distance_m"]

    for c in required_aoa:
        if c not in aoa_df.columns:
            raise ValueError(f"Missing AoA column: {c}")

    for c in required_dist:
        if c not in dist_df.columns:
            raise ValueError(f"Missing distance column: {c}")

    df = pd.merge(
        aoa_df[["group", "true_angle_deg", "pred_angle_deg"]],
        dist_df[["group", "true_distance_m", "pred_distance_m"]],
        on="group",
        how="inner"
    )

    if len(df) == 0:
        raise ValueError("Hybrid merge empty. Check group names.")

    for c in ["true_angle_deg", "pred_angle_deg", "true_distance_m", "pred_distance_m"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna()

    df["x_true"], df["y_true"] = angle_distance_to_xy(
        df["true_angle_deg"].values,
        df["true_distance_m"].values
    )

    df["x_pred"], df["y_pred"] = angle_distance_to_xy(
        df["pred_angle_deg"].values,
        df["pred_distance_m"].values
    )

    df["error_2d_m"] = np.sqrt(
        (df["x_true"] - df["x_pred"]) ** 2 +
        (df["y_true"] - df["y_pred"]) ** 2
    )

    df["model"] = "Proposed_Hybrid"

    hybrid_pred_path = os.path.join(
        OUT_DIR,
        "proposed_hybrid_filelevel_predictions.csv"
    )

    df.to_csv(hybrid_pred_path, index=False)

    print("\nSaved proposed hybrid predictions:")
    print(hybrid_pred_path, flush=True)

    return compute_metrics(df, "Proposed_Hybrid")


# ============================================================
# IEEE BAR GRAPHS
# ============================================================

def plot_metric_bar(results_df, metric_col, ylabel, title, filename, lower_is_better=True):
    df = results_df.copy()
    df = df.sort_values(metric_col, ascending=lower_is_better)

    models = df["model"].values
    values = df[metric_col].values

    plt.figure(figsize=(10, 5))

    bars = plt.bar(models, values, edgecolor="black")

    for bar, v in zip(bars, values):
        if "percent" in metric_col:
            text = f"{v:.1f}%"
        elif "r2" in metric_col:
            text = f"{v:.3f}"
        else:
            text = f"{v:.3f}"

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.xticks(rotation=25, ha="right")
    plt.xlabel("Model")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()

    out_path = os.path.join(GRAPH_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved bar graph:", out_path, flush=True)


def save_bar_comparison_graphs(results_df):

    plot_metric_bar(
        results_df,
        metric_col="rmse_2d_m",
        ylabel="RMSE 2D (m)",
        title="2D Localization RMSE Comparison",
        filename="bar_rmse_2d.png",
        lower_is_better=True
    )

    plot_metric_bar(
        results_df,
        metric_col="within_1m_percent",
        ylabel="Samples Within 1 m (%)",
        title="Localization Accuracy Within 1 m",
        filename="bar_within_1m.png",
        lower_is_better=False
    )

    plot_metric_bar(
        results_df,
        metric_col="mean_2d_error_m",
        ylabel="Mean 2D Error (m)",
        title="Mean 2D Localization Error Comparison",
        filename="bar_mean_2d_error.png",
        lower_is_better=True
    )

    plot_metric_bar(
        results_df,
        metric_col="within_2m_percent",
        ylabel="Samples Within 2 m (%)",
        title="Localization Accuracy Within 2 m",
        filename="bar_within_2m.png",
        lower_is_better=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n===================================================")
    print(" FINAL 2D LOCALIZATION PERFORMANCE COMPARISON")
    print(" Direct ML Models + Proposed File-Level Hybrid")
    print(" WITH IEEE BAR GRAPHS")
    print("===================================================\n", flush=True)

    train_df = ensure_xy(read_csv(TRAIN_FILE))
    val_df   = ensure_xy(read_csv(VAL_FILE))
    test_df  = ensure_xy(read_csv(TEST_FILE))

    feature_cols = select_features(train_df)

    print(f"\nTotal selected direct-model features: {len(feature_cols)}", flush=True)

    train_df = train_df.dropna(subset=["x_true", "y_true"])
    val_df   = val_df.dropna(subset=["x_true", "y_true"])
    test_df  = test_df.dropna(subset=["x_true", "y_true"])

    X_train = train_df[feature_cols]
    y_train = train_df[["x_true", "y_true"]].values

    X_test = test_df[feature_cols]

    results = []
    all_predictions = []

    models = get_direct_models()

    # ========================================================
    # DIRECT MODEL COMPARISON
    # ========================================================

    for model_name, model in models.items():

        print(f"\nTraining direct 2D model: {model_name}", flush=True)

        model.fit(X_train, y_train)

        test_pred = model.predict(X_test)

        pred_df = test_df.copy()
        pred_df[f"{model_name}_x_pred"] = test_pred[:, 0]
        pred_df[f"{model_name}_y_pred"] = test_pred[:, 1]

        file_df = file_level_from_window_predictions(pred_df, model_name)
        file_df["model"] = model_name

        metrics = compute_metrics(file_df, model_name)
        results.append(metrics)

        all_predictions.append(file_df)

        model_path = os.path.join(MODEL_DIR, f"direct_2d_{model_name}.pkl")

        joblib.dump({
            "model": model,
            "features": feature_cols,
            "target_cols": ["x_true", "y_true"],
            "metrics": metrics
        }, model_path)

        print(f"{model_name} RMSE 2D : {metrics['rmse_2d_m']:.6f} m", flush=True)
        print(f"{model_name} Mean Error : {metrics['mean_2d_error_m']:.6f} m", flush=True)
        print(f"{model_name} Within 1m : {metrics['within_1m_percent']:.2f}%", flush=True)

    # ========================================================
    # PROPOSED HYBRID FILE-LEVEL MODEL
    # ========================================================

    print("\nRunning proposed file-level Hybrid fusion...", flush=True)

    hybrid_metrics = run_filelevel_hybrid()
    results.append(hybrid_metrics)

    print(f"Hybrid RMSE 2D : {hybrid_metrics['rmse_2d_m']:.6f} m", flush=True)
    print(f"Hybrid Mean Error : {hybrid_metrics['mean_2d_error_m']:.6f} m", flush=True)
    print(f"Hybrid Within 1m : {hybrid_metrics['within_1m_percent']:.2f}%", flush=True)

    # ========================================================
    # SAVE FINAL COMPARISON
    # ========================================================

    results_df = pd.DataFrame(results).sort_values("rmse_2d_m")

    comparison_path = os.path.join(
        OUT_DIR,
        "final_2d_performance_comparison.csv"
    )

    results_df.to_csv(comparison_path, index=False)

    # Save direct model predictions
    if len(all_predictions) > 0:
        direct_pred_df = pd.concat(all_predictions, ignore_index=True)

        direct_pred_path = os.path.join(
            OUT_DIR,
            "direct_models_filelevel_predictions.csv"
        )

        direct_pred_df.to_csv(direct_pred_path, index=False)

    # Save IEEE bar graphs
    save_bar_comparison_graphs(results_df)

    print("\n================ FINAL PERFORMANCE COMPARISON ================")
    print(results_df, flush=True)

    print("\nSaved comparison table:")
    print(comparison_path, flush=True)

    print("\nSaved graphs in:")
    print(GRAPH_DIR, flush=True)

    print("\nGenerated IEEE bar graph files:")
    print(" - bar_rmse_2d.png")
    print(" - bar_within_1m.png")
    print(" - bar_mean_2d_error.png")
    print(" - bar_within_2m.png")

    print("\nFor IEEE paper, mainly use:")
    print(" 1) bar_rmse_2d.png")
    print(" 2) bar_within_1m.png")
    print(" 3) bar_mean_2d_error.png optional")

    print("\nPerformance comparison completed successfully.", flush=True)


if __name__ == "__main__":
    main()