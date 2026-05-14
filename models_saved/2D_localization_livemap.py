#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import mode
from collections import Counter

# ============================================================
# USER SETTINGS
# ============================================================

FREQ_FOLDER = "850MHz"
FREQ_MHZ = 850

DIST_FOLDER = "2m"

# Use only angle folder that exists for all four pairs
ANGLE_FOLDER = "30"

# Your file name is like rep1_rx.dat
REP_KEYWORD = "rep1_rx"

WINDOW_SIZE = 8192
STEP_SIZE = 4096
MAX_RANGE = 5

DATA_ROOT = "data/raw_iq/rx"

AOA_MODEL_PATH = "models_saved/fast_extratrees_aoa.pkl"
DIST_MODEL_PATH = "models_saved/xgboost_rssi_distance.pkl"

OUT_DIR = "results/live_filelevel_2d_localization"
os.makedirs(OUT_DIR, exist_ok=True)

PAIR_FOLDERS = {
    "pair13": "(1,3)",
    "pair14": "(1,4)",
    "pair23": "(2,3)",
    "pair24": "(2,4)",
}

PAIR_NAMES = ["pair13", "pair14", "pair23", "pair24"]


# ============================================================
# FILE FINDING
# ============================================================

def find_iq_file(pair_folder):
    search_dir = os.path.join(
        DATA_ROOT,
        FREQ_FOLDER,
        DIST_FOLDER,
        pair_folder,
        ANGLE_FOLDER
    )

    if not os.path.exists(search_dir):
        raise FileNotFoundError(
            f"\nFolder not found:\n{search_dir}\n\n"
            f"Check available folders using:\n"
            f'ls "data/raw_iq/rx/{FREQ_FOLDER}/{DIST_FOLDER}/{pair_folder}"'
        )

    csv_files = glob.glob(os.path.join(search_dir, "*.csv"))
    dat_files = glob.glob(os.path.join(search_dir, "*.dat"))

    files = csv_files + dat_files

    if len(files) == 0:
        raise FileNotFoundError(f"No .csv or .dat files found in: {search_dir}")

    rep_matches = [
        f for f in files
        if REP_KEYWORD.lower() in os.path.basename(f).lower()
    ]

    if len(rep_matches) > 0:
        selected = rep_matches[0]
    else:
        selected = files[0]

    print(f"Selected file for {pair_folder}: {selected}")
    return selected


# ============================================================
# IQ LOADING
# ============================================================

def load_iq_file(path):
    """
    Supports:
    1. CSV with columns I1,Q1,I2,Q2
    2. CSV with 4 columns without exact headers
    3. DAT float32 format: I1,Q1,I2,Q2,I1,Q1,I2,Q2,...
    """

    if path.endswith(".csv"):
        df = pd.read_csv(path)

        required = ["I1", "Q1", "I2", "Q2"]

        if all(c in df.columns for c in required):
            x1 = df["I1"].values + 1j * df["Q1"].values
            x2 = df["I2"].values + 1j * df["Q2"].values
            return x1, x2

        if df.shape[1] >= 4:
            x1 = df.iloc[:, 0].values + 1j * df.iloc[:, 1].values
            x2 = df.iloc[:, 2].values + 1j * df.iloc[:, 3].values
            return x1, x2

        raise ValueError(f"CSV must contain I1,Q1,I2,Q2 or 4 columns: {path}")

    elif path.endswith(".dat"):
        raw = np.fromfile(path, dtype=np.float32)

        if len(raw) < 4:
            raise ValueError(f"DAT file too small: {path}")

        if len(raw) % 4 != 0:
            raw = raw[:len(raw) - (len(raw) % 4)]

        raw = raw.reshape(-1, 4)

        x1 = raw[:, 0] + 1j * raw[:, 1]
        x2 = raw[:, 2] + 1j * raw[:, 3]

        return x1, x2

    else:
        raise ValueError(f"Unsupported file format: {path}")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def remove_dc(x):
    return x - np.mean(x)


def wrap_phase(x):
    return np.arctan2(np.sin(x), np.cos(x))


def extract_pair_features(x1, x2, pair_name, freq_mhz):
    x1 = remove_dc(x1)
    x2 = remove_dc(x2)

    power1 = np.mean(np.abs(x1) ** 2)
    power2 = np.mean(np.abs(x2) ** 2)

    rssi1 = 10 * np.log10(power1 + 1e-12)
    rssi2 = 10 * np.log10(power2 + 1e-12)

    phase_diff = np.angle(x1 * np.conj(x2))
    unwrap_phase = np.unwrap(phase_diff)

    R11 = np.mean(x1 * np.conj(x1))
    R12 = np.mean(x1 * np.conj(x2))
    R22 = np.mean(x2 * np.conj(x2))

    corr_peak = np.abs(np.vdot(x1, x2)) / (
        np.sqrt(np.sum(np.abs(x1) ** 2) * np.sum(np.abs(x2) ** 2)) + 1e-12
    )

    cov_phase = np.angle(R12)

    c = 3e8
    wavelength = c / (freq_mhz * 1e6)

    # If your actual antenna spacing is known, replace this.
    # For 850 MHz, lambda/2 ≈ 0.176 m.
    d = wavelength / 2

    phase_med = np.median(phase_diff)
    val = (phase_med * wavelength) / (2 * np.pi * d)
    val = np.clip(val, -1, 1)
    aoa_physics = np.rad2deg(np.arcsin(val))

    f = {}

    f[f"{pair_name}_power1"] = power1
    f[f"{pair_name}_power2"] = power2
    f[f"{pair_name}_rssi1"] = rssi1
    f[f"{pair_name}_rssi2"] = rssi2
    f[f"{pair_name}_rssi_mean"] = (rssi1 + rssi2) / 2
    f[f"{pair_name}_rssi_diff"] = rssi1 - rssi2

    f[f"{pair_name}_power_ratio"] = power1 / (power2 + 1e-12)
    f[f"{pair_name}_power_diff"] = power1 - power2

    f[f"{pair_name}_phase_diff_mean"] = np.mean(phase_diff)
    f[f"{pair_name}_phase_diff_median"] = np.median(phase_diff)
    f[f"{pair_name}_phase_diff_std"] = np.std(phase_diff)

    f[f"{pair_name}_unwrap_phase_mean"] = np.mean(unwrap_phase)
    f[f"{pair_name}_unwrap_phase_median"] = np.median(unwrap_phase)
    f[f"{pair_name}_unwrap_phase_std"] = np.std(unwrap_phase)

    f[f"{pair_name}_R11_real"] = np.real(R11)
    f[f"{pair_name}_R12_real"] = np.real(R12)
    f[f"{pair_name}_R12_imag"] = np.imag(R12)
    f[f"{pair_name}_R22_real"] = np.real(R22)

    f[f"{pair_name}_cov_phase"] = cov_phase
    f[f"{pair_name}_corr_peak"] = corr_peak
    f[f"{pair_name}_aoa_physics_deg"] = aoa_physics

    return f


def add_phase_extra_features(row):
    row = row.copy()

    for p in PAIR_NAMES:
        phase_cols = [
            f"{p}_phase_diff_mean",
            f"{p}_phase_diff_median",
            f"{p}_unwrap_phase_mean",
            f"{p}_unwrap_phase_median",
            f"{p}_cov_phase",
        ]

        for col in phase_cols:
            if col in row:
                val = row[col]
                w = wrap_phase(val)
                row[f"{col}_wrap"] = w
                row[f"{col}_sin"] = np.sin(w)
                row[f"{col}_cos"] = np.cos(w)

        rcol = f"{p}_R12_real"
        icol = f"{p}_R12_imag"

        if rcol in row and icol in row:
            phase = np.arctan2(row[icol], row[rcol])
            row[f"{p}_R12_phase"] = phase
            row[f"{p}_R12_phase_sin"] = np.sin(phase)
            row[f"{p}_R12_phase_cos"] = np.cos(phase)
            row[f"{p}_R12_mag"] = np.sqrt(row[rcol] ** 2 + row[icol] ** 2)

    return row


def prepare_model_input(row, feature_cols):
    data = {}

    for col in feature_cols:
        data[col] = row.get(col, np.nan)

    return pd.DataFrame([data])


def angle_distance_to_xy(angle_deg, distance_m):
    theta = np.deg2rad(angle_deg)
    x = distance_m * np.sin(theta)
    y = distance_m * np.cos(theta)
    return x, y


# ============================================================
# RADAR PLOT
# ============================================================

def plot_file_level_radar(final_angle, final_distance, x_pred, y_pred,
                          window_df, true_angle=None, true_distance=None):

    fig, ax = plt.subplots(figsize=(8, 8))

    theta = np.linspace(-90, 90, 300)

    for r in range(1, MAX_RANGE + 1):
        x = r * np.sin(np.deg2rad(theta))
        y = r * np.cos(np.deg2rad(theta))
        ax.plot(x, y, "--", linewidth=0.8, alpha=0.4)
        ax.text(0.05, r, f"{r} m", fontsize=8)

    for ang in [-60, -45, -30, -15, 0, 15, 30, 45, 60]:
        x = MAX_RANGE * np.sin(np.deg2rad(ang))
        y = MAX_RANGE * np.cos(np.deg2rad(ang))

        ax.plot([0, x], [0, y], ":", linewidth=0.8, alpha=0.5)

        ax.text(
            (MAX_RANGE + 0.25) * np.sin(np.deg2rad(ang)),
            (MAX_RANGE + 0.25) * np.cos(np.deg2rad(ang)),
            f"{ang}°",
            fontsize=8,
            ha="center",
            va="center"
        )

    # Receiver
    ax.scatter(0, 0, marker="s", s=120, label="Receiver Array")

    # Window-level noisy estimates as small points
    ax.scatter(
        window_df["x_window"],
        window_df["y_window"],
        marker=".",
        s=18,
        alpha=0.35,
        label="Window-level estimates"
    )

    # Final file-level prediction
    ax.scatter(
        x_pred,
        y_pred,
        marker="x",
        s=180,
        linewidths=3,
        label="Final File-level Predicted TX"
    )

    ax.plot([0, x_pred], [0, y_pred], linewidth=2.0, alpha=0.8)

    # Optional true point from folder name
    if true_angle is not None and true_distance is not None:
        x_true, y_true = angle_distance_to_xy(true_angle, true_distance)

        ax.scatter(
            x_true,
            y_true,
            marker="o",
            s=120,
            label="Ground Truth TX"
        )

        ax.plot(
            [x_true, x_pred],
            [y_true, y_pred],
            linewidth=1.2,
            alpha=0.7
        )

        err = np.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2)

        title = (
            "File-level 2D Localization Radar Map\n"
            f"GT: θ={true_angle:.1f}°, d={true_distance:.2f} m | "
            f"Pred: θ={final_angle:.1f}°, d={final_distance:.2f} m | "
            f"Error={err:.2f} m"
        )
    else:
        title = (
            "File-level 2D Localization Radar Map\n"
            f"Pred: θ={final_angle:.1f}°, d={final_distance:.2f} m"
        )

    ax.text(
        0.02,
        0.96,
        f"File-level AoA: {final_angle:.1f}°\n"
        f"File-level distance: {final_distance:.2f} m\n"
        f"X: {x_pred:.2f} m\n"
        f"Y: {y_pred:.2f} m\n"
        f"Windows used: {len(window_df)}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", alpha=0.15)
    )

    ax.set_xlim(-MAX_RANGE - 0.5, MAX_RANGE + 0.5)
    ax.set_ylim(-0.5, MAX_RANGE + 0.8)

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")

    plt.tight_layout()

    out_png = os.path.join(
        OUT_DIR,
        f"filelevel_radar_{FREQ_FOLDER}_{DIST_FOLDER}_{ANGLE_FOLDER}.png"
    )

    plt.savefig(out_png, dpi=300)
    plt.show()

    print("\nSaved file-level radar map:")
    print(out_png)


# ============================================================
# FILE-LEVEL AGGREGATION
# ============================================================

def majority_vote(values):
    counts = Counter(values)
    return counts.most_common(1)[0][0]


def confidence_weighted_vote(angle_values, conf_values):
    scores = {}

    for a, c in zip(angle_values, conf_values):
        scores[a] = scores.get(a, 0.0) + c

    return max(scores, key=scores.get)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n===================================================")
    print(" FILE-LEVEL 2D LOCALIZATION RADAR MAP")
    print("===================================================\n")

    print("Configuration:")
    print("Frequency :", FREQ_FOLDER)
    print("Distance  :", DIST_FOLDER)
    print("Angle     :", ANGLE_FOLDER)
    print("Rep       :", REP_KEYWORD)

    print("\nFinding IQ files...")

    pair_files = {}
    for pair_name, folder_name in PAIR_FOLDERS.items():
        pair_files[pair_name] = find_iq_file(folder_name)

    print("\nLoading trained models...")

    aoa_bundle = joblib.load(AOA_MODEL_PATH)
    dist_bundle = joblib.load(DIST_MODEL_PATH)

    aoa_model = aoa_bundle["model"]
    aoa_encoder = aoa_bundle["label_encoder"]
    aoa_features = aoa_bundle["feature_cols"]

    dist_model = dist_bundle["model"]
    dist_features = dist_bundle["feature_cols"]

    print("AoA model loaded      :", AOA_MODEL_PATH)
    print("Distance model loaded :", DIST_MODEL_PATH)

    print("\nLoading IQ data...")

    iq_data = {}
    for pair_name, path in pair_files.items():
        iq_data[pair_name] = load_iq_file(path)

    min_len = min(len(v[0]) for v in iq_data.values())

    print(f"\nMinimum IQ samples available: {min_len}")
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Step size: {STEP_SIZE}")
    print(f"Frequency: {FREQ_MHZ} MHz")

    if min_len < WINDOW_SIZE:
        raise ValueError("IQ file length is smaller than WINDOW_SIZE.")

    rows = []

    print("\nProcessing windows for file-level prediction...\n")

    for start in range(0, min_len - WINDOW_SIZE, STEP_SIZE):
        row = {"frequency_mhz": FREQ_MHZ}

        for pair_name, (x1_all, x2_all) in iq_data.items():
            x1 = x1_all[start:start + WINDOW_SIZE]
            x2 = x2_all[start:start + WINDOW_SIZE]

            feats = extract_pair_features(x1, x2, pair_name, FREQ_MHZ)
            row.update(feats)

        row = add_phase_extra_features(row)

        # AoA prediction
        X_aoa = prepare_model_input(row, aoa_features)
        aoa_pred_enc = aoa_model.predict(X_aoa)
        aoa_pred = float(aoa_encoder.inverse_transform(aoa_pred_enc)[0])

        aoa_prob = aoa_model.predict_proba(X_aoa)
        aoa_conf = float(np.max(aoa_prob))

        # Distance prediction
        X_dist = prepare_model_input(row, dist_features)
        dist_pred = float(dist_model.predict(X_dist)[0])
        dist_pred = max(dist_pred, 0.0)

        x_window, y_window = angle_distance_to_xy(aoa_pred, dist_pred)

        rows.append({
            "window_index": start // STEP_SIZE,
            "start_sample": start,
            "angle_pred_deg": aoa_pred,
            "angle_confidence": aoa_conf,
            "distance_pred_m": dist_pred,
            "x_window": x_window,
            "y_window": y_window,
        })

        print(
            f"Window {start // STEP_SIZE:04d} | "
            f"AoA={aoa_pred:6.1f}° | "
            f"d={dist_pred:5.2f} m | "
            f"x={x_window:6.2f} m | "
            f"y={y_window:6.2f} m | "
            f"conf={aoa_conf:.2f}"
        )

    window_df = pd.DataFrame(rows)

    if len(window_df) == 0:
        raise ValueError("No windows processed.")

    # ========================================================
    # FILE-LEVEL FINAL DECISION
    # ========================================================

    final_angle = confidence_weighted_vote(
        window_df["angle_pred_deg"].values,
        window_df["angle_confidence"].values
    )

    final_distance = float(np.median(window_df["distance_pred_m"].values))

    x_pred, y_pred = angle_distance_to_xy(final_angle, final_distance)

    print("\n================ FILE-LEVEL FINAL RESULT ================")
    print(f"Final AoA       : {final_angle:.2f} degrees")
    print(f"Final distance  : {final_distance:.3f} m")
    print(f"Final X         : {x_pred:.3f} m")
    print(f"Final Y         : {y_pred:.3f} m")
    print(f"Windows used    : {len(window_df)}")

    # Save window-level predictions
    window_csv = os.path.join(
        OUT_DIR,
        f"window_predictions_{FREQ_FOLDER}_{DIST_FOLDER}_{ANGLE_FOLDER}.csv"
    )
    window_df.to_csv(window_csv, index=False)

    print("\nSaved window prediction CSV:")
    print(window_csv)

    # Save final result
    result_df = pd.DataFrame([{
        "frequency_mhz": FREQ_MHZ,
        "distance_folder": DIST_FOLDER,
        "angle_folder": ANGLE_FOLDER,
        "rep_keyword": REP_KEYWORD,
        "final_angle_deg": final_angle,
        "final_distance_m": final_distance,
        "final_x_m": x_pred,
        "final_y_m": y_pred,
        "num_windows": len(window_df),
        "mean_window_confidence": window_df["angle_confidence"].mean(),
    }])

    result_csv = os.path.join(
        OUT_DIR,
        f"filelevel_result_{FREQ_FOLDER}_{DIST_FOLDER}_{ANGLE_FOLDER}.csv"
    )
    result_df.to_csv(result_csv, index=False)

    print("\nSaved final file-level result:")
    print(result_csv)

    # Ground truth from folders
    try:
        true_distance = float(DIST_FOLDER.lower().replace("m", ""))
        true_angle = float(ANGLE_FOLDER)
    except Exception:
        true_distance = None
        true_angle = None

    if true_distance is not None and true_angle is not None:
        x_true, y_true = angle_distance_to_xy(true_angle, true_distance)
        err = np.sqrt((x_pred - x_true) ** 2 + (y_pred - y_true) ** 2)

        print("\n================ GROUND TRUTH COMPARISON ================")
        print(f"True AoA       : {true_angle:.2f} degrees")
        print(f"True distance  : {true_distance:.3f} m")
        print(f"True X         : {x_true:.3f} m")
        print(f"True Y         : {y_true:.3f} m")
        print(f"2D error       : {err:.3f} m")

    # Plot final radar map
    plot_file_level_radar(
        final_angle=final_angle,
        final_distance=final_distance,
        x_pred=x_pred,
        y_pred=y_pred,
        window_df=window_df,
        true_angle=true_angle,
        true_distance=true_distance,
    )

    print("\nFile-level radar localization completed successfully.")


if __name__ == "__main__":
    main()