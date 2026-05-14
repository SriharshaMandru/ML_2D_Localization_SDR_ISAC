#!/usr/bin/env python3
import os
import pandas as pd

POS_FILE = "data/processed/final_merged_four_pair_dataset_positive.csv"
NEG_FILE = "data/processed/final_merged_four_pair_dataset_negative_augmented.csv"

OUT_FILE = "data/processed/final_merged_four_pair_dataset_augmented.csv"

SORT_COLS = [
    "frequency_mhz",
    "distance_m",
    "angle_deg",
    "rep",
    "window_id",
]


def main():
    if not os.path.exists(POS_FILE):
        raise FileNotFoundError(f"Positive file not found: {POS_FILE}")

    if not os.path.exists(NEG_FILE):
        raise FileNotFoundError(f"Negative file not found: {NEG_FILE}")

    pos_df = pd.read_csv(POS_FILE)
    neg_df = pd.read_csv(NEG_FILE)

    # Add augmented flag if missing
    if "augmented" not in pos_df.columns:
        pos_df["augmented"] = 0

    if "augmented" not in neg_df.columns:
        neg_df["augmented"] = 1

    print("Positive dataset shape:", pos_df.shape)
    print("Negative dataset shape:", neg_df.shape)

    print("\nPositive angle distribution:")
    print(pos_df["angle_deg"].value_counts().sort_index())

    print("\nNegative angle distribution:")
    print(neg_df["angle_deg"].value_counts().sort_index())

    final_df = pd.concat([pos_df, neg_df], ignore_index=True)

    sort_cols = [c for c in SORT_COLS if c in final_df.columns]
    final_df = final_df.sort_values(sort_cols).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    final_df.to_csv(OUT_FILE, index=False)

    print("\nFinal dataset saved:")
    print(OUT_FILE)

    print("\nFinal shape:", final_df.shape)

    print("\nFinal angle distribution:")
    print(final_df["angle_deg"].value_counts().sort_index())

    print("\nAugmented distribution:")
    print(final_df["augmented"].value_counts().sort_index())

    print("\nPreview:")
    print(final_df.head())


if __name__ == "__main__":
    main()