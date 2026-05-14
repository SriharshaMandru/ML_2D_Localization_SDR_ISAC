import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.utils.model_utils import numeric_feature_columns
from scripts.utils.sign_correct_aoa import apply_sign_correction_batch

DATASET_FILE = "data/processed/final_dataset_split.csv"
MODEL_FILE = "models_saved/angle_branch_extratrees.pkl"
PRED_FILE = "results/predictions/angle_predictions.csv"
TABLE_FILE = "results/tables/angle_error_table.csv"


def main():
    df = pd.read_csv(DATASET_FILE)

    target = "target_angle_deg"

    drop_cols = [
        "split",
        "angle_deg",
        "distance_m",
        "target_angle_deg",
        "target_distance_m",
        "target_x",
        "target_y",
    ]

    feature_cols = numeric_feature_columns(df, drop_cols)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    model = ExtraTreesRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        max_features="sqrt",
    )

    model.fit(train_df[feature_cols], train_df[target])

    for name, part in [("VALIDATION", val_df), ("TEST", test_df)]:
        pred = model.predict(part[feature_cols])
        pred_signed = apply_sign_correction_batch(pred, part)

        print(f"\n{name} AoA RESULTS")
        print("--- Raw model output (magnitude only) ---")
        print("MAE :", mean_absolute_error(part[target], pred))
        print("RMSE:", mean_squared_error(part[target], pred) ** 0.5)
        print("R2  :", r2_score(part[target], pred))
        print("--- Sign-corrected output (for signed AoA inference) ---")
        print("MAE :", mean_absolute_error(part[target], pred_signed))
        print("RMSE:", mean_squared_error(part[target], pred_signed) ** 0.5)
        print("R2  :", r2_score(part[target], pred_signed))

    test_pred = model.predict(test_df[feature_cols])
    # Sign-corrected predictions: magnitude from model, sign from phase diff
    test_pred_signed = apply_sign_correction_batch(test_pred, test_df)

    os.makedirs("models_saved", exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "target": target,
        },
        MODEL_FILE,
    )

    os.makedirs("results/predictions", exist_ok=True)

    pred_df = test_df[
        [
            "frequency_mhz",
            "distance_m",
            "angle_deg",
            "rep",
            "window_id",
        ]
    ].copy()

    pred_df["true_angle_deg"] = test_df[target].values
    pred_df["pred_angle_deg"] = test_pred
    pred_df["pred_angle_deg_signed"] = test_pred_signed
    pred_df["angle_error_deg"] = pred_df["pred_angle_deg"] - pred_df["true_angle_deg"]
    pred_df["angle_error_signed_deg"] = pred_df["pred_angle_deg_signed"] - pred_df["true_angle_deg"]
    pred_df["abs_angle_error_deg"] = np.abs(pred_df["angle_error_deg"])
    pred_df["abs_angle_error_signed_deg"] = np.abs(pred_df["angle_error_signed_deg"])

    pred_df.to_csv(PRED_FILE, index=False)

    os.makedirs("results/tables", exist_ok=True)

    summary = pred_df.groupby(["frequency_mhz", "distance_m"]).agg(
        samples=("abs_angle_error_deg", "count"),
        mae_deg=("abs_angle_error_deg", "mean"),
        rmse_deg=("angle_error_deg", lambda x: np.sqrt(np.mean(x ** 2))),
        bias_deg=("angle_error_deg", "mean"),
    ).reset_index()

    summary.to_csv(TABLE_FILE, index=False)

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv("results/tables/angle_feature_importance.csv", index=False)

    print("\nSaved:", MODEL_FILE)
    print("Saved:", PRED_FILE)
    print("Saved:", TABLE_FILE)


if __name__ == "__main__":
    main()
