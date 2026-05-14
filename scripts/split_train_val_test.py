#!/usr/bin/env python3
import os
import pandas as pd

INPUT_FILE = "data/processed/final_dataset_split_ml_safe.csv"

TRAIN_FILE = "data/processed/train_dataset.csv"
VAL_FILE   = "data/processed/val_dataset.csv"
TEST_FILE  = "data/processed/test_dataset.csv"

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "split" not in df.columns:
        raise ValueError("split column missing")

    # Split
    train_df = df[df["split"] == "train"].copy()
    val_df   = df[df["split"] == "val"].copy()
    test_df  = df[df["split"] == "test"].copy()

    # Remove split column after separation
    train_df = train_df.drop(columns=["split"])
    val_df   = val_df.drop(columns=["split"])
    test_df  = test_df.drop(columns=["split"])

    os.makedirs("data/processed", exist_ok=True)

    train_df.to_csv(TRAIN_FILE, index=False)
    val_df.to_csv(VAL_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print("Saved:")
    print(TRAIN_FILE, train_df.shape)
    print(VAL_FILE, val_df.shape)
    print(TEST_FILE, test_df.shape)

    print("\nTrain angle distribution:")
    print(train_df["angle_deg"].value_counts().sort_index())

    print("\nValidation angle distribution:")
    print(val_df["angle_deg"].value_counts().sort_index())

    print("\nTest angle distribution:")
    print(test_df["angle_deg"].value_counts().sort_index())

    print("\nColumns in train file:")
    print(train_df.columns.tolist())

if __name__ == "__main__":
    main()