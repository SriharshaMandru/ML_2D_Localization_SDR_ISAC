#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = "data/processed/train_dataset.csv"
VAL_FILE   = "data/processed/val_dataset.csv"
TEST_FILE  = "data/processed/test_dataset.csv"

OUT_DIR = "results/aoa_model_comparison_music_ml"
GRAPH_DIR = os.path.join(OUT_DIR, "graphs")
MODEL_DIR = "models_saved"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ANGLE_COL = "angle_deg"
RANDOM_STATE = 42

PAIR_PREFIXES = ["pair13", "pair14", "pair23", "pair24"]
ANGLE_CLASSES = np.array([-60, -45, -30, -15, 0, 15, 30, 45, 60], dtype=float)


# ============================================================
# BASIC HELPERS
# ============================================================

def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {path}: {df.shape}")
    return df


def wrap_phase(x):
    return np.arctan2(np.sin(x), np.cos(x))


def nearest_angle(theta):
    return ANGLE_CLASSES[np.argmin(np.abs(ANGLE_CLASSES - theta))]


def get_group_cols(df):
    if "rx_file_path" in df.columns:
        return ["rx_file_path"]
    elif "rx_file_name" in df.columns:
        return ["rx_file_name"]
    else:
        return [
            c for c in ["frequency_mhz", "freq_mhz", "distance_m", ANGLE_COL, "rep"]
            if c in df.columns
        ]


# ============================================================
# PHASE FEATURE ENGINEERING
# ============================================================

def add_phase_features(df):
    df = df.copy()

    for p in PAIR_PREFIXES:
        phase_cols = [
            f"{p}_phase_diff_mean",
            f"{p}_phase_diff_median",
            f"{p}_unwrap_phase_mean",
            f"{p}_unwrap_phase_median",
            f"{p}_cov_phase",
        ]

        for col in phase_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[f"{col}_wrap"] = wrap_phase(df[col])
                df[f"{col}_sin"] = np.sin(df[f"{col}_wrap"])
                df[f"{col}_cos"] = np.cos(df[f"{col}_wrap"])

        rcol = f"{p}_R12_real"
        icol = f"{p}_R12_imag"

        if rcol in df.columns and icol in df.columns:
            df[rcol] = pd.to_numeric(df[rcol], errors="coerce")
            df[icol] = pd.to_numeric(df[icol], errors="coerce")

            df[f"{p}_R12_phase"] = np.arctan2(df[icol], df[rcol])
            df[f"{p}_R12_phase_sin"] = np.sin(df[f"{p}_R12_phase"])
            df[f"{p}_R12_phase_cos"] = np.cos(df[f"{p}_R12_phase"])
            df[f"{p}_R12_mag"] = np.sqrt(df[rcol] ** 2 + df[icol] ** 2)

    return df


def select_candidate_features(df):
    exclude_keywords = [
        "angle_deg",
        "distance_m",
        "x_true",
        "y_true",
        "x_pred",
        "y_pred",
        "split",
        "file",
        "path",
        "name",
        "rep",
        "tx_available",
    ]

    features = []

    for col in df.columns:
        low = col.lower()

        if any(k in low for k in exclude_keywords):
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            features.append(col)

    return features


# ============================================================
# MUSIC BASELINE
# ============================================================

def music_single_pair_aoa(row, pair_prefix, angle_grid=np.linspace(-90, 90, 361)):
    r11_col = f"{pair_prefix}_R11_real"
    r12r_col = f"{pair_prefix}_R12_real"
    r12i_col = f"{pair_prefix}_R12_imag"
    r22_col = f"{pair_prefix}_R22_real"

    if not all(c in row.index for c in [r11_col, r12r_col, r12i_col, r22_col]):
        return np.nan

    try:
        R11 = float(row[r11_col])
        R12 = float(row[r12r_col]) + 1j * float(row[r12i_col])
        R22 = float(row[r22_col])

        if np.isnan(R11) or np.isnan(np.real(R12)) or np.isnan(np.imag(R12)) or np.isnan(R22):
            return np.nan

        R = np.array([
            [R11, R12],
            [np.conj(R12), R22]
        ], dtype=complex)

        _, eigvecs = np.linalg.eigh(R)

        En = eigvecs[:, 0].reshape(-1, 1)

        pseudospectrum = []

        for theta in angle_grid:
            phase = np.pi * np.sin(np.deg2rad(theta))
            a = np.array([1.0, np.exp(-1j * phase)], dtype=complex).reshape(-1, 1)

            denom = np.conj(a).T @ En @ np.conj(En).T @ a
            p = 1.0 / (np.abs(denom[0, 0]) + 1e-12)
            pseudospectrum.append(p)

        best_theta = angle_grid[int(np.argmax(pseudospectrum))]
        return nearest_angle(best_theta)

    except Exception:
        return np.nan


def music_predict_dataframe(df):
    preds = []

    for _, row in df.iterrows():
        pair_preds = []

        for p in PAIR_PREFIXES:
            pred = music_single_pair_aoa(row, p)
            if not np.isnan(pred):
                pair_preds.append(pred)

        if len(pair_preds) == 0:
            preds.append(np.nan)
        else:
            preds.append(nearest_angle(np.median(pair_preds)))

    return np.array(preds, dtype=float)


# ============================================================
# FILE-LEVEL AGGREGATION
# ============================================================

def file_level_from_window_predictions(df, pred_col, conf_col=None):
    group_cols = get_group_cols(df)
    rows = []

    for key, g in df.groupby(group_cols):
        true_angle = float(g[ANGLE_COL].iloc[0])

        if conf_col is not None and conf_col in g.columns:
            scores = {}
            for _, row in g.iterrows():
                pred = float(row[pred_col])
                conf = float(row[conf_col])
                scores[pred] = scores.get(pred, 0.0) + conf
            final_pred = max(scores, key=scores.get)
        else:
            final_pred = float(g[pred_col].mode().iloc[0])

        rows.append({
            "group": str(key),
            "true_angle_deg": true_angle,
            "pred_angle_deg": float(final_pred),
            "abs_error_deg": abs(float(final_pred) - true_angle),
            "num_windows": len(g),
        })

    return pd.DataFrame(rows)


def evaluate_window_and_file(df_test, pred_col, model_name, conf_col=None):
    temp = df_test.dropna(subset=[pred_col]).copy()

    y_true = temp[ANGLE_COL].astype(float)
    y_pred = temp[pred_col].astype(float)

    win_acc = accuracy_score(y_true, y_pred)
    win_mae = mean_absolute_error(y_true, y_pred)
    win_mse = mean_squared_error(y_true, y_pred)
    win_rmse = np.sqrt(win_mse)
    win_r2 = r2_score(y_true, y_pred)

    file_df = file_level_from_window_predictions(temp, pred_col, conf_col)

    fy_true = file_df["true_angle_deg"].astype(float)
    fy_pred = file_df["pred_angle_deg"].astype(float)

    file_acc = accuracy_score(fy_true, fy_pred)
    file_mae = mean_absolute_error(fy_true, fy_pred)
    file_mse = mean_squared_error(fy_true, fy_pred)
    file_rmse = np.sqrt(file_mse)
    file_r2 = r2_score(fy_true, fy_pred)

    full_result = {
        "model": model_name,
        "window_accuracy_percent": win_acc * 100,
        "window_mae_deg": win_mae,
        "window_mse_deg2": win_mse,
        "window_rmse_deg": win_rmse,
        "window_r2": win_r2,
        "file_accuracy_percent": file_acc * 100,
        "file_mae_deg": file_mae,
        "file_mse_deg2": file_mse,
        "file_rmse_deg": file_rmse,
        "file_r2": file_r2,
        "num_test_windows": len(temp),
        "num_test_files": len(file_df),
    }

    paper_result = {
        "model": model_name,
        "accuracy_percent": file_acc * 100,
        "mae_deg": file_mae,
        "rmse_deg": file_rmse,
        "r2": file_r2,
    }

    return full_result, paper_result, file_df


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models():
    models = {}

    models["ExtraTrees"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ))
    ])

    models["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ))
    ])

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=2,
            ))
        ])

    models["MLP"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=300,
            random_state=RANDOM_STATE,
            early_stopping=True,
        ))
    ])

    return models


# ============================================================
# XGBOOST-ONLY PLOTS
# ============================================================

def save_xgboost_confusion_matrix(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title("XGBoost AoA Confusion Matrix")
    plt.xlabel("Predicted AoA (degrees)")
    plt.ylabel("True AoA (degrees)")
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)
    plt.colorbar(label="Number of Windows")

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()


def save_xgboost_true_vs_predicted_scatter(y_true, y_pred, out_path):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.45)

    min_angle = min(np.min(y_true), np.min(y_pred))
    max_angle = max(np.max(y_true), np.max(y_pred))

    plt.plot(
        [min_angle, max_angle],
        [min_angle, max_angle],
        "--",
        linewidth=1.5
    )

    plt.xlabel("True AoA (degrees)")
    plt.ylabel("Predicted AoA (degrees)")
    plt.title("XGBoost True vs Predicted AoA")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(ANGLE_CLASSES)
    plt.yticks(ANGLE_CLASSES)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n===================================================")
    print(" AoA COMPARISON: MUSIC vs ML MODELS")
    print(" Final table: Accuracy, MAE, RMSE, R²")
    print(" XGBoost-only confusion matrix and scatter plot")
    print("===================================================\n")

    train_df = read_csv(TRAIN_FILE)
    val_df = read_csv(VAL_FILE)
    test_df = read_csv(TEST_FILE)

    train_df = add_phase_features(train_df)
    val_df = add_phase_features(val_df)
    test_df = add_phase_features(test_df)

    feature_cols = select_candidate_features(train_df)
    print(f"\nTotal candidate ML features: {len(feature_cols)}")

    train_full_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    X_train = train_full_df[feature_cols]
    y_train = train_full_df[ANGLE_COL].astype(float)

    X_test = test_df[feature_cols]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    labels = le.classes_

    full_results = []
    paper_results = []
    file_outputs = {}

    # ========================================================
    # MUSIC BASELINE
    # ========================================================

    print("\nRunning MUSIC baseline on test data...")
    test_df["MUSIC_pred"] = music_predict_dataframe(test_df)

    music_full_result, music_paper_result, music_file_df = evaluate_window_and_file(
        test_df,
        pred_col="MUSIC_pred",
        model_name="MUSIC",
        conf_col=None
    )

    full_results.append(music_full_result)
    paper_results.append(music_paper_result)
    file_outputs["MUSIC"] = music_file_df

    print("\nMUSIC paper results:")
    print(music_paper_result)

    # ========================================================
    # ML MODELS
    # ========================================================

    models = build_models()

    for name, model in models.items():
        print(f"\nTraining {name} AoA model...")

        model.fit(X_train, y_train_enc)

        pred_enc = model.predict(X_test)
        pred_deg = le.inverse_transform(pred_enc)

        test_df[f"{name}_pred"] = pred_deg

        # XGBoost-only plots immediately after XGBoost prediction
        if name == "XGBoost":
            save_xgboost_confusion_matrix(
                test_df[ANGLE_COL].astype(float),
                test_df["XGBoost_pred"].astype(float),
                labels,
                os.path.join(GRAPH_DIR, "xgboost_confusion_matrix.png")
            )

            save_xgboost_true_vs_predicted_scatter(
                test_df[ANGLE_COL].astype(float),
                test_df["XGBoost_pred"].astype(float),
                os.path.join(GRAPH_DIR, "xgboost_true_vs_predicted_scatter.png")
            )

        if hasattr(model.named_steps["clf"], "predict_proba"):
            prob = model.predict_proba(X_test)
            test_df[f"{name}_conf"] = np.max(prob, axis=1)
            conf_col = f"{name}_conf"
        else:
            test_df[f"{name}_conf"] = 1.0
            conf_col = f"{name}_conf"

        full_result, paper_result, file_df = evaluate_window_and_file(
            test_df,
            pred_col=f"{name}_pred",
            model_name=name,
            conf_col=conf_col
        )

        full_results.append(full_result)
        paper_results.append(paper_result)
        file_outputs[name] = file_df

        print(f"\n{name} paper results:")
        print(paper_result)

        model_path = os.path.join(MODEL_DIR, f"aoa_compare_{name}.pkl")
        joblib.dump({
            "model": model,
            "label_encoder": le,
            "feature_cols": feature_cols,
            "angle_classes": labels.tolist(),
        }, model_path)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    full_results_df = pd.DataFrame(full_results).sort_values("file_mae_deg")
    paper_results_df = pd.DataFrame(paper_results).sort_values("mae_deg")

    full_path = os.path.join(OUT_DIR, "aoa_model_comparison_full_metrics.csv")
    paper_path = os.path.join(OUT_DIR, "aoa_model_comparison_results.csv")

    full_results_df.to_csv(full_path, index=False)
    paper_results_df.to_csv(paper_path, index=False)

    print("\n================ FINAL AoA COMPARISON ================")
    print(paper_results_df)

    print("\nSaved final paper comparison table:")
    print(paper_path)

    print("\nSaved full metrics table:")
    print(full_path)

    for name, file_df in file_outputs.items():
        file_path = os.path.join(OUT_DIR, f"{name}_file_level_predictions.csv")
        file_df.to_csv(file_path, index=False)

    window_path = os.path.join(OUT_DIR, "aoa_window_predictions_all_models.csv")
    test_df.to_csv(window_path, index=False)

    print("\nSaved all window predictions:")
    print(window_path)

    print("\nSaved XGBoost-only graphs:")
    print(os.path.join(GRAPH_DIR, "xgboost_confusion_matrix.png"))
    print(os.path.join(GRAPH_DIR, "xgboost_true_vs_predicted_scatter.png"))

    print("\nAoA comparison completed successfully.")


if __name__ == "__main__":
    main()