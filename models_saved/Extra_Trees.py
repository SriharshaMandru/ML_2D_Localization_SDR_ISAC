#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# ============================================================
# LIMIT CPU THREADS
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    classification_report,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = "data/processed/train_dataset.csv"
VAL_FILE   = "data/processed/val_dataset.csv"
TEST_FILE  = "data/processed/test_dataset.csv"

OUT_DIR = "results/aoa_fast_extratrees"
GRAPH_DIR = os.path.join(OUT_DIR, "graphs")
MODEL_DIR = "models_saved"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ANGLE_COL = "angle_deg"
RANDOM_STATE = 42

PAIR_PREFIXES = ["pair13", "pair14", "pair23", "pair24"]


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)
    print(f"Loaded {path}: {df.shape}", flush=True)
    return df


def wrap_phase(x):
    return np.arctan2(np.sin(x), np.cos(x))


# ============================================================
# FEATURE ENGINEERING
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


# ============================================================
# FEATURE SELECTION
# ============================================================

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

    if len(features) == 0:
        raise ValueError("No features selected. Check dataset columns.")

    return features


# ============================================================
# MODEL
# ============================================================

def train_model(X_train, y_train_enc, n_estimators=250):
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=1,
        )),
    ])

    model.fit(X_train, y_train_enc)
    return model


def top_feature_selection(train_df, feature_cols, y_train_enc, top_k=50):
    print("\nTraining feature-importance model...", flush=True)

    base_model = train_model(
        train_df[feature_cols],
        y_train_enc,
        n_estimators=180
    )

    clf = base_model.named_steps["clf"]

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)

    imp_path = os.path.join(OUT_DIR, "all_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)

    print("Saved feature importance:", imp_path, flush=True)

    selected = imp_df.head(top_k)["feature"].tolist()

    return selected, imp_df


# ============================================================
# FILE LEVEL VOTING
# ============================================================

def file_level_voting(pred_df):
    if "rx_file_path" in pred_df.columns:
        group_cols = ["rx_file_path"]
    elif "rx_file_name" in pred_df.columns:
        group_cols = ["rx_file_name"]
    else:
        group_cols = [
            c for c in ["frequency_mhz", "distance_m", ANGLE_COL, "rep"]
            if c in pred_df.columns
        ]

    if not group_cols:
        print("No file/group column found. Skipping file-level voting.", flush=True)
        return None

    rows = []

    for key, g in pred_df.groupby(group_cols):
        true_angle = g[ANGLE_COL].iloc[0]

        scores = {}

        for _, row in g.iterrows():
            pred = row["angle_pred_deg"]
            conf = row["angle_confidence"]
            scores[pred] = scores.get(pred, 0.0) + conf

        final_angle = max(scores, key=scores.get)

        rows.append({
            "group": str(key),
            "true_angle_deg": true_angle,
            "pred_angle_deg": final_angle,
            "avg_confidence": g["angle_confidence"].mean(),
            "num_windows": len(g),
            "abs_error_deg": abs(final_angle - true_angle),
        })

    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def save_confusion_matrix(y_true, y_pred, labels, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("AoA Confusion Matrix - ExtraTrees")
    plt.xlabel("Predicted AoA (degrees)")
    plt.ylabel("True AoA (degrees)")
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)
    plt.colorbar(label="Number of Samples")

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=10
            )

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_true_vs_predicted_scatter(y_true, y_pred, path):
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

    min_angle = min(y_true.min(), y_pred.min())
    max_angle = max(y_true.max(), y_pred.max())

    plt.plot(
        [min_angle, max_angle],
        [min_angle, max_angle],
        "k--",
        linewidth=1.5,
        label="Ideal Prediction"
    )

    plt.xlabel("True AoA (degrees)")
    plt.ylabel("Predicted AoA (degrees)")
    plt.title("True vs Predicted AoA - ExtraTrees")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_angle_mae_bar(pred_df, path):
    mae_df = pred_df.groupby(ANGLE_COL)["angle_abs_error_deg"].mean()

    plt.figure(figsize=(7, 4))
    mae_df.plot(kind="bar", edgecolor="black")

    plt.xlabel("True AoA (degrees)")
    plt.ylabel("Mean Absolute Error (degrees)")
    plt.title("Angle-wise AoA MAE - ExtraTrees")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n===================================================")
    print(" FAST EXTRA TREES AoA MODEL WITH IEEE GRAPHS")
    print("===================================================\n", flush=True)

    train_df = read_csv(TRAIN_FILE)
    val_df = read_csv(VAL_FILE)
    test_df = read_csv(TEST_FILE)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if ANGLE_COL not in df.columns:
            raise ValueError(f"{ANGLE_COL} missing in {name} dataset")

    print("\nAdding phase features...", flush=True)

    train_df = add_phase_features(train_df)
    val_df = add_phase_features(val_df)
    test_df = add_phase_features(test_df)

    feature_cols = select_candidate_features(train_df)

    print(f"\nCandidate features selected: {len(feature_cols)}", flush=True)

    y_train = train_df[ANGLE_COL].astype(float)
    y_val = val_df[ANGLE_COL].astype(float)
    y_test = test_df[ANGLE_COL].astype(float)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    labels = le.classes_

    print("Angle classes:", labels, flush=True)

    best_features, imp_df = top_feature_selection(
        train_df=train_df,
        feature_cols=feature_cols,
        y_train_enc=y_train_enc,
        top_k=50
    )

    selected_path = os.path.join(OUT_DIR, "selected_top50_features.txt")

    with open(selected_path, "w") as f:
        for c in best_features:
            f.write(c + "\n")

    print("\nSelected Top 50 features saved:", selected_path, flush=True)

    print("\nTraining final ExtraTrees AoA model...", flush=True)

    final_model = train_model(
        train_df[best_features],
        y_train_enc,
        n_estimators=250
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    val_pred_enc = final_model.predict(val_df[best_features])
    val_pred = le.inverse_transform(val_pred_enc)

    print("\n================ VALIDATION RESULTS ================", flush=True)
    print(f"Validation Accuracy : {accuracy_score(y_val, val_pred) * 100:.2f}%", flush=True)
    print(f"Validation MAE      : {mean_absolute_error(y_val, val_pred):.3f} degrees", flush=True)
    print(classification_report(y_val, val_pred), flush=True)

    # ========================================================
    # TEST
    # ========================================================

    test_pred_enc = final_model.predict(test_df[best_features])
    test_pred = le.inverse_transform(test_pred_enc)

    test_prob = final_model.predict_proba(test_df[best_features])
    test_conf = np.max(test_prob, axis=1)

    print("\n================ TEST WINDOW RESULTS ================", flush=True)
    print(f"Test Accuracy : {accuracy_score(y_test, test_pred) * 100:.2f}%", flush=True)
    print(f"Test MAE      : {mean_absolute_error(y_test, test_pred):.3f} degrees", flush=True)
    print(classification_report(y_test, test_pred), flush=True)

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    pred_df = test_df.copy()
    pred_df["angle_pred_deg"] = test_pred
    pred_df["angle_confidence"] = test_conf
    pred_df["angle_abs_error_deg"] = np.abs(
        pred_df[ANGLE_COL] - pred_df["angle_pred_deg"]
    )

    pred_path = os.path.join(OUT_DIR, "aoa_window_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    print("\nSaved window predictions:", pred_path, flush=True)

    # ========================================================
    # FILE LEVEL RESULT
    # ========================================================

    file_df = file_level_voting(pred_df)

    if file_df is not None:
        file_path = os.path.join(OUT_DIR, "aoa_file_level_predictions.csv")
        file_df.to_csv(file_path, index=False)

        file_acc = accuracy_score(
            file_df["true_angle_deg"],
            file_df["pred_angle_deg"]
        )

        file_mae = mean_absolute_error(
            file_df["true_angle_deg"],
            file_df["pred_angle_deg"]
        )

        print("\n================ FILE LEVEL RESULTS ================", flush=True)
        print(f"File-level Accuracy : {file_acc * 100:.2f}%", flush=True)
        print(f"File-level MAE      : {file_mae:.3f} degrees", flush=True)
        print("Saved file-level predictions:", file_path, flush=True)

    # ========================================================
    # SAVE GRAPHS
    # ========================================================

    cm_path = os.path.join(GRAPH_DIR, "aoa_confusion_matrix.png")
    scatter_path = os.path.join(GRAPH_DIR, "true_vs_predicted_aoa_scatter.png")
    mae_path = os.path.join(GRAPH_DIR, "angle_wise_mae.png")

    save_confusion_matrix(y_test, test_pred, labels, cm_path)
    save_true_vs_predicted_scatter(y_test, test_pred, scatter_path)
    save_angle_mae_bar(pred_df, mae_path)

    print("\nSaved confusion matrix:", cm_path, flush=True)
    print("Saved True vs Predicted scatter graph:", scatter_path, flush=True)
    print("Saved angle-wise MAE graph:", mae_path, flush=True)

    # ========================================================
    # SAVE MODEL
    # ========================================================

    bundle = {
        "model": final_model,
        "label_encoder": le,
        "feature_cols": best_features,
        "angle_classes": labels.tolist(),
    }

    model_path = os.path.join(MODEL_DIR, "fast_extratrees_aoa.pkl")
    joblib.dump(bundle, model_path)

    print("\nSaved model:", model_path, flush=True)
    print("\nAoA training and graph generation completed successfully.", flush=True)


if __name__ == "__main__":
    main()