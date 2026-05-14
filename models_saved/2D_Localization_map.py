#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_FILE = "results/final_2d_localization/final_2d_localization_predictions.csv"

OUT_DIR = "results/final_2d_localization/graphs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FILE = os.path.join(OUT_DIR, "ieee_2d_radar_localization_map.png")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_cols = [
    "x_true", "y_true",
    "x_pred", "y_pred",
    "true_angle_deg",
    "pred_angle_deg",
    "true_distance_m",
    "pred_distance_m",
    "error_2d_m"
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")


# ============================================================
# RADAR MAP SETTINGS
# ============================================================

max_range = max(
    df["true_distance_m"].max(),
    df["pred_distance_m"].max()
)

radar_limit = np.ceil(max_range + 0.5)

theta = np.linspace(-90, 90, 300)

fig, ax = plt.subplots(figsize=(8, 8))

# ============================================================
# DRAW RADAR RANGE CIRCLES
# ============================================================

for r in np.arange(1, radar_limit + 1, 1):
    x_circle = r * np.sin(np.deg2rad(theta))
    y_circle = r * np.cos(np.deg2rad(theta))

    ax.plot(x_circle, y_circle, linestyle="--", linewidth=0.8, alpha=0.5)
    ax.text(0.05, r, f"{r:.0f} m", fontsize=9)


# ============================================================
# DRAW ANGLE LINES
# ============================================================

angle_lines = [-60, -45, -30, -15, 0, 15, 30, 45, 60]

for ang in angle_lines:
    x_line = radar_limit * np.sin(np.deg2rad(ang))
    y_line = radar_limit * np.cos(np.deg2rad(ang))

    ax.plot([0, x_line], [0, y_line], linestyle=":", linewidth=0.8, alpha=0.6)

    label_x = (radar_limit + 0.25) * np.sin(np.deg2rad(ang))
    label_y = (radar_limit + 0.25) * np.cos(np.deg2rad(ang))

    ax.text(label_x, label_y, f"{ang}°", fontsize=9, ha="center", va="center")


# ============================================================
# RECEIVER ARRAY POSITION
# ============================================================

ax.scatter(0, 0, marker="s", s=120, label="Receiver Array / Radar Origin")
ax.text(0.05, -0.15, "RX Array", fontsize=10)


# ============================================================
# PLOT TRUE AND PREDICTED POSITIONS
# ============================================================

ax.scatter(
    df["x_true"],
    df["y_true"],
    marker="o",
    s=70,
    label="Ground Truth TX"
)

ax.scatter(
    df["x_pred"],
    df["y_pred"],
    marker="x",
    s=80,
    label="Predicted TX"
)


# ============================================================
# DRAW ERROR LINES
# ============================================================

for _, row in df.iterrows():
    ax.plot(
        [row["x_true"], row["x_pred"]],
        [row["y_true"], row["y_pred"]],
        linewidth=0.8,
        alpha=0.45
    )


# ============================================================
# OPTIONAL: SHOW ERROR VALUE NEAR PREDICTED POINT
# ============================================================

for _, row in df.iterrows():
    ax.text(
        row["x_pred"] + 0.03,
        row["y_pred"] + 0.03,
        f"{row['error_2d_m']:.2f}m",
        fontsize=7,
        alpha=0.75
    )


# ============================================================
# AXIS SETTINGS
# ============================================================

ax.set_xlim(-radar_limit - 0.5, radar_limit + 0.5)
ax.set_ylim(-0.5, radar_limit + 0.7)

ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title("Radar-Style 2D Localization Map\nAoA + RSSI Distance Fusion")

ax.grid(True, linestyle="--", alpha=0.4)
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=300)
plt.show()

print("Radar-style 2D localization map saved to:")
print(OUT_FILE)