#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# INPUT FILES
# ============================================================

AOA_FILE = "results/aoa_fast_extratrees/aoa_file_level_predictions.csv"
DIST_FILE = "results/xgboost_rssi_distance/distance_file_level_predictions.csv"

OUT_DIR = "results/final_2d_localization"
GRAPH_DIR = os.path.join(OUT_DIR, "graphs")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# FUNCTIONS
# ============================================================

def read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    print(f"Loaded {path}: {df.shape}")
    return df


def angle_distance_to_xy(angle_deg, distance_m):
    """
    0 degree = straight forward / broadside.
    +angle = right side.
    -angle = left side.
    """
    theta = np.deg2rad(angle_deg)
    x = distance_m * np.sin(theta)
    y = distance_m * np.cos(theta)
    return x, y


def compute_metrics(df):
    mean_error = df["error_2d_m"].mean()
    median_error = df["error_2d_m"].median()
    rmse_error = np.sqrt(np.mean(df["error_2d_m"] ** 2))
    max_error = df["error_2d_m"].max()

    mean_x_error = df["x_error_m"].abs().mean()
    mean_y_error = df["y_error_m"].abs().mean()

    return {
        "mean_2d_error_m": mean_error,
        "median_2d_error_m": median_error,
        "rmse_2d_error_m": rmse_error,
        "max_2d_error_m": max_error,
        "mean_abs_x_error_m": mean_x_error,
        "mean_abs_y_error_m": mean_y_error,
    }


# ============================================================
# IEEE PLOTS
# ============================================================

def plot_true_vs_pred_xy(df, path):
    plt.figure(figsize=(8, 7))

    plt.scatter(
        df["x_true"],
        df["y_true"],
        marker="o",
        label="Ground Truth"
    )

    plt.scatter(
        df["x_pred"],
        df["y_pred"],
        marker="x",
        label="Predicted"
    )

    for _, row in df.iterrows():
        plt.plot(
            [row["x_true"], row["x_pred"]],
            [row["y_true"], row["y_pred"]],
            linewidth=0.7,
            alpha=0.5,
        )

    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("2D Localization: Ground Truth vs Predicted")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_error_histogram(df, path):
    plt.figure(figsize=(7, 5))

    plt.hist(
        df["error_2d_m"],
        bins=20,
        edgecolor="black"
    )

    plt.xlabel("2D Localization Error (m)")
    plt.ylabel("Number of Test Files")
    plt.title("2D Localization Error Distribution")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_cdf_error(df, path):
    """
    CDF of 2D localization error.
    Important IEEE-standard graph for localization papers.
    """

    errors = np.sort(df["error_2d_m"].values)
    cdf = np.arange(1, len(errors) + 1) / len(errors)

    plt.figure(figsize=(7, 5))

    plt.plot(
        errors,
        cdf,
        linewidth=2
    )

    plt.xlabel("2D Localization Error (m)")
    plt.ylabel("CDF")
    plt.title("CDF of 2D Localization Error")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_error_vs_distance(df, path):
    plt.figure(figsize=(7, 5))

    plt.scatter(
        df["true_distance_m"],
        df["error_2d_m"],
        alpha=0.75,
        edgecolors="black",
        linewidths=0.4
    )

    plt.xlabel("True Distance (m)")
    plt.ylabel("2D Error (m)")
    plt.title("2D Error vs True Distance")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_error_vs_angle(df, path):
    plt.figure(figsize=(7, 5))

    plt.scatter(
        df["true_angle_deg"],
        df["error_2d_m"],
        alpha=0.75,
        edgecolors="black",
        linewidths=0.4
    )

    plt.xlabel("True AoA (degree)")
    plt.ylabel("2D Error (m)")
    plt.title("2D Error vs AoA")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_true_pred_distance(df, path):
    plt.figure(figsize=(7, 6))

    plt.scatter(
        df["true_distance_m"],
        df["pred_distance_m"],
        alpha=0.75,
        edgecolors="black",
        linewidths=0.4
    )

    min_v = min(
        df["true_distance_m"].min(),
        df["pred_distance_m"].min()
    )

    max_v = max(
        df["true_distance_m"].max(),
        df["pred_distance_m"].max()
    )

    plt.plot(
        [min_v, max_v],
        [min_v, max_v],
        "k--",
        linewidth=1.5,
        label="Ideal Prediction"
    )

    plt.xlabel("True Distance (m)")
    plt.ylabel("Predicted Distance (m)")
    plt.title("True vs Predicted Distance")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_true_pred_angle(df, path):
    plt.figure(figsize=(7, 6))

    plt.scatter(
        df["true_angle_deg"],
        df["pred_angle_deg"],
        alpha=0.75,
        edgecolors="black",
        linewidths=0.4
    )

    min_v = min(
        df["true_angle_deg"].min(),
        df["pred_angle_deg"].min()
    )

    max_v = max(
        df["true_angle_deg"].max(),
        df["pred_angle_deg"].max()
    )

    plt.plot(
        [min_v, max_v],
        [min_v, max_v],
        "k--",
        linewidth=1.5,
        label="Ideal Prediction"
    )

    plt.xlabel("Ground Truth AoA (degree)")
    plt.ylabel("Predicted AoA (degree)")
    plt.title("True vs Predicted AoA")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_2d_error_bar_by_distance(df, path):
    """
    Distance-wise mean 2D localization error.
    Useful for IEEE result discussion.
    """

    err_df = df.groupby("true_distance_m")["error_2d_m"].mean()

    plt.figure(figsize=(7, 4))

    err_df.plot(
        kind="bar",
        edgecolor="black"
    )

    plt.xlabel("True Distance (m)")
    plt.ylabel("Mean 2D Error (m)")
    plt.title("Distance-wise Mean 2D Localization Error")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_2d_error_bar_by_angle(df, path):
    """
    AoA-wise mean 2D localization error.
    Useful for IEEE result discussion.
    """

    err_df = df.groupby("true_angle_deg")["error_2d_m"].mean()

    plt.figure(figsize=(7, 4))

    err_df.plot(
        kind="bar",
        edgecolor="black"
    )

    plt.xlabel("True AoA (degree)")
    plt.ylabel("Mean 2D Error (m)")
    plt.title("AoA-wise Mean 2D Localization Error")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n===================================================")
    print(" FINAL 2D LOCALIZATION FUSION")
    print(" AoA + RSSI Distance → X,Y")
    print(" WITH IEEE-STANDARD GRAPHS")
    print("===================================================\n")

    aoa_df = read_csv(AOA_FILE)
    dist_df = read_csv(DIST_FILE)

    # --------------------------------------------------------
    # Merge using group column
    # --------------------------------------------------------

    if "group" not in aoa_df.columns or "group" not in dist_df.columns:
        raise ValueError(
            "Both AoA and distance prediction files must contain 'group' column."
        )

    df = pd.merge(
        aoa_df,
        dist_df,
        on="group",
        how="inner",
        suffixes=("_aoa", "_dist")
    )

    print(f"Merged final dataset: {df.shape}")

    if len(df) == 0:
        raise ValueError(
            "Merged dataset is empty. Check whether group names match."
        )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required = [
        "true_angle_deg",
        "pred_angle_deg",
        "true_distance_m",
        "pred_distance_m",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column after merge: {col}")

    # --------------------------------------------------------
    # True x,y from ground truth AoA + ground truth distance
    # --------------------------------------------------------

    df["x_true"], df["y_true"] = angle_distance_to_xy(
        df["true_angle_deg"].astype(float),
        df["true_distance_m"].astype(float),
    )

    # --------------------------------------------------------
    # Predicted x,y from predicted AoA + predicted distance
    # --------------------------------------------------------

    df["x_pred"], df["y_pred"] = angle_distance_to_xy(
        df["pred_angle_deg"].astype(float),
        df["pred_distance_m"].astype(float),
    )

    # --------------------------------------------------------
    # Error calculation
    # --------------------------------------------------------

    df["x_error_m"] = df["x_pred"] - df["x_true"]
    df["y_error_m"] = df["y_pred"] - df["y_true"]

    df["error_2d_m"] = np.sqrt(
        df["x_error_m"] ** 2 + df["y_error_m"] ** 2
    )

    df["angle_abs_error_deg"] = np.abs(
        df["pred_angle_deg"] - df["true_angle_deg"]
    )

    df["distance_abs_error_m"] = np.abs(
        df["pred_distance_m"] - df["true_distance_m"]
    )

    # --------------------------------------------------------
    # Save final prediction file
    # --------------------------------------------------------

    final_path = os.path.join(
        OUT_DIR,
        "final_2d_localization_predictions.csv"
    )

    df.to_csv(final_path, index=False)

    print("\nSaved final 2D localization predictions:")
    print(final_path)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = compute_metrics(df)

    metrics["mean_angle_error_deg"] = df["angle_abs_error_deg"].mean()
    metrics["mean_distance_error_m"] = df["distance_abs_error_m"].mean()
    metrics["num_test_files"] = len(df)

    metrics_df = pd.DataFrame([metrics])

    metrics_path = os.path.join(
        OUT_DIR,
        "final_2d_metrics_summary.csv"
    )

    metrics_df.to_csv(metrics_path, index=False)

    print("\n================ FINAL 2D LOCALIZATION RESULTS ================")
    print(f"Number of test files      : {metrics['num_test_files']}")
    print(f"Mean AoA error            : {metrics['mean_angle_error_deg']:.3f} degrees")
    print(f"Mean distance error       : {metrics['mean_distance_error_m']:.3f} m")
    print(f"Mean 2D error             : {metrics['mean_2d_error_m']:.3f} m")
    print(f"Median 2D error           : {metrics['median_2d_error_m']:.3f} m")
    print(f"RMSE 2D error             : {metrics['rmse_2d_error_m']:.3f} m")
    print(f"Maximum 2D error          : {metrics['max_2d_error_m']:.3f} m")
    print(f"Mean absolute X error     : {metrics['mean_abs_x_error_m']:.3f} m")
    print(f"Mean absolute Y error     : {metrics['mean_abs_y_error_m']:.3f} m")

    print("\nSaved metrics summary:")
    print(metrics_path)

    # --------------------------------------------------------
    # Save IEEE-style plots
    # --------------------------------------------------------

    plot_true_vs_pred_xy(
        df,
        os.path.join(GRAPH_DIR, "ieee_true_vs_predicted_xy.png")
    )

    plot_error_histogram(
        df,
        os.path.join(GRAPH_DIR, "ieee_2d_error_histogram.png")
    )

    plot_cdf_error(
        df,
        os.path.join(GRAPH_DIR, "ieee_2d_error_cdf.png")
    )

    plot_error_vs_distance(
        df,
        os.path.join(GRAPH_DIR, "ieee_2d_error_vs_distance.png")
    )

    plot_error_vs_angle(
        df,
        os.path.join(GRAPH_DIR, "ieee_2d_error_vs_angle.png")
    )

    plot_true_pred_distance(
        df,
        os.path.join(GRAPH_DIR, "ieee_true_vs_predicted_distance.png")
    )

    plot_true_pred_angle(
        df,
        os.path.join(GRAPH_DIR, "ieee_true_vs_predicted_aoa.png")
    )

    plot_2d_error_bar_by_distance(
        df,
        os.path.join(GRAPH_DIR, "ieee_distance_wise_2d_error.png")
    )

    plot_2d_error_bar_by_angle(
        df,
        os.path.join(GRAPH_DIR, "ieee_aoa_wise_2d_error.png")
    )

    print("\nSaved IEEE-style graphs in:")
    print(GRAPH_DIR)

    print("\nGenerated graph files:")
    print(" - ieee_true_vs_predicted_xy.png")
    print(" - ieee_2d_error_histogram.png")
    print(" - ieee_2d_error_cdf.png")
    print(" - ieee_2d_error_vs_distance.png")
    print(" - ieee_2d_error_vs_angle.png")
    print(" - ieee_true_vs_predicted_distance.png")
    print(" - ieee_true_vs_predicted_aoa.png")
    print(" - ieee_distance_wise_2d_error.png")
    print(" - ieee_aoa_wise_2d_error.png")

    print("\nFinal 2D localization fusion completed successfully.")


if __name__ == "__main__":
    main()