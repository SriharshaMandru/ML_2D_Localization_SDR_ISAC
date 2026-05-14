#!/usr/bin/env python3
import os
import pandas as pd

INPUT_FILE = "data/processed/pairwise_features.csv"
OUT_FILE = "data/processed/pairwise_features_augmented.csv"

SIGN_FLIP_KEYWORDS = ["phase", "aoa", "imag"]

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "angle_deg" not in df.columns:
        raise ValueError("Missing column: angle_deg")

    df["augmented"] = 0

    # Do not duplicate 0 degree
    aug = df[df["angle_deg"] != 0].copy()

    aug["angle_deg"] = -aug["angle_deg"]

    for col in aug.columns:
        if any(k in col.lower() for k in SIGN_FLIP_KEYWORDS):
            if pd.api.types.is_numeric_dtype(aug[col]):
                aug[col] = -aug[col]

    aug["augmented"] = 1

    final_df = pd.concat([df, aug], ignore_index=True)

    sort_cols = [
        "frequency_mhz",
        "distance_m",
        "angle_deg",
        "pair",
        "rep",
        "window_id",
        "augmented",
    ]
    sort_cols = [c for c in sort_cols if c in final_df.columns]

    final_df = final_df.sort_values(sort_cols, ignore_index=True)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    final_df.to_csv(OUT_FILE, index=False)

    print("Saved:", OUT_FILE)
    print("Shape:", final_df.shape)

    print("\nAngle distribution:")
    print(final_df["angle_deg"].value_counts().sort_index())

    print("\nAugmented distribution:")
    print(final_df["augmented"].value_counts().sort_index())

if __name__ == "__main__":
    main()