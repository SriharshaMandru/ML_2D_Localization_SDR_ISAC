#!/usr/bin/env python3
import os
import pandas as pd

INPUT_FILE = "data/processed/final_merged_four_pair_dataset_positive.csv"
OUT_FILE = "data/processed/final_merged_four_pair_dataset_negative_augmented.csv"

SIGN_FLIP_KEYWORDS = [
    "phase",
    "aoa",
    "imag",
]

SORT_COLS = [
    "frequency_mhz",
    "distance_m",
    "angle_deg",
    "rep",
    "window_id",
]


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "angle_deg" not in df.columns:
        raise ValueError("ERROR: angle_deg column missing")

    print("Loaded positive dataset:")
    print(INPUT_FILE)
    print("Shape:", df.shape)

    print("\nPositive angle distribution:")
    print(df["angle_deg"].value_counts().sort_index())

    # Only non-zero positive angles are mirrored
    # 0 degree is skipped because -0 = 0
    neg_df = df[df["angle_deg"] != 0].copy()

    # Mirror angle label
    neg_df["angle_deg"] = -neg_df["angle_deg"]

    # Mark as augmented data
    neg_df["augmented"] = 1

    # Flip phase/AoA/imaginary-related numeric features
    for col in neg_df.columns:
        col_lower = col.lower()

        if any(k in col_lower for k in SIGN_FLIP_KEYWORDS):
            if pd.api.types.is_numeric_dtype(neg_df[col]):
                neg_df[col] = -neg_df[col]

    sort_cols = [c for c in SORT_COLS if c in neg_df.columns]
    neg_df = neg_df.sort_values(sort_cols).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    neg_df.to_csv(OUT_FILE, index=False)

    print("\nNegative augmented dataset saved:")
    print(OUT_FILE)

    print("\nNegative augmented shape:", neg_df.shape)

    print("\nNegative angle distribution:")
    print(neg_df["angle_deg"].value_counts().sort_index())

    print("\nPreview:")
    print(neg_df.head())


if __name__ == "__main__":
    main()