import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.utils.model_utils import distance_feature_columns

try:
    from xgboost import XGBRegressor
    USE_XGBOOST = True
except Exception:
    USE_XGBOOST = False


DATASET_FILE = "data/processed/final_dataset_split.csv"
MODEL_FILE = "models_saved/distance_branch_xgboost.pkl"
PRED_FILE = "results/predictions/distance_predictions.csv"
TABLE_FILE = "results/tables/distance_error_table.csv"


def main():
    df = pd.read_csv(DATASET_FILE)

    target = "target_distance_m"
    feature_cols = distance_feature_columns(df)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    if USE_XGBOOST:
        model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    else:
        print("XGBoost not installed. Using RandomForestRegressor.")
        model = RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        )

    model.fit(train_df[feature_cols], train_df[target])

    for name, part in [("VALIDATION", val_df), ("TEST", test_df)]:
        pred = model.predict(part[feature_cols])

        print(f"\n{name} DISTANCE RESULTS")
        print("MAE :", mean_absolute_error(part[target], pred))
        print("RMSE:", mean_squared_error(part[target], pred) ** 0.5)
        print("R2  :", r2_score(part[target], pred))

    test_pred = model.predict(test_df[feature_cols])

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

    pred_df["true_distance_m"] = test_df[target].values
    pred_df["pred_distance_m"] = test_pred
    pred_df["distance_error_m"] = pred_df["pred_distance_m"] - pred_df["true_distance_m"]
    pred_df["abs_distance_error_m"] = np.abs(pred_df["distance_error_m"])

    pred_df.to_csv(PRED_FILE, index=False)

    os.makedirs("results/tables", exist_ok=True)

    summary = pred_df.groupby(["frequency_mhz", "distance_m"]).agg(
        samples=("abs_distance_error_m", "count"),
        mae_m=("abs_distance_error_m", "mean"),
        rmse_m=("distance_error_m", lambda x: np.sqrt(np.mean(x ** 2))),
        bias_m=("distance_error_m", "mean"),
    ).reset_index()

    summary.to_csv(TABLE_FILE, index=False)

    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        importance.to_csv("results/tables/distance_feature_importance.csv", index=False)

    print("\nSaved:", MODEL_FILE)
    print("Saved:", PRED_FILE)
    print("Saved:", TABLE_FILE)


if __name__ == "__main__":
    main()
