import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scripts.utils.model_utils import numeric_feature_columns

DATASET_FILE = "data/processed/final_dataset_split.csv"
MODEL_FILE = "models_saved/direct_xy_regression.pkl"
PRED_FILE = "results/predictions/direct_xy_regression_xy_predictions.csv"


def main():
    df = pd.read_csv(DATASET_FILE)

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
    test_df = df[df["split"] == "test"]

    X_train = train_df[feature_cols]
    y_train = train_df[["target_x", "target_y"]]

    X_test = test_df[feature_cols]
    y_test = test_df[["target_x", "target_y"]]

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=700, random_state=42, n_jobs=-1, min_samples_leaf=1))
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    err = np.sqrt(
        (pred[:, 0] - y_test["target_x"].values) ** 2
        + (pred[:, 1] - y_test["target_y"].values) ** 2
    )

    print("\nDirect XY Regression XY RESULTS")
    print("Mean loc error:", np.mean(err))
    print("Median loc error:", np.median(err))
    print("P90 loc error:", np.percentile(err, 90))
    print("RMSE XY:", mean_squared_error(y_test, pred) ** 0.5)
    print("MAE XY:", mean_absolute_error(y_test, pred))
    print("R2 X:", r2_score(y_test["target_x"], pred[:, 0]))
    print("R2 Y:", r2_score(y_test["target_y"], pred[:, 1]))

    os.makedirs("models_saved", exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
        },
        MODEL_FILE,
    )

    os.makedirs("results/predictions", exist_ok=True)

    out = test_df[["frequency_mhz", "distance_m", "angle_deg", "rep", "window_id"]].copy()
    out["true_x"] = y_test["target_x"].values
    out["true_y"] = y_test["target_y"].values
    out["pred_x"] = pred[:, 0]
    out["pred_y"] = pred[:, 1]
    out["localization_error_m"] = err
    out.to_csv(PRED_FILE, index=False)

    print("Saved:", MODEL_FILE)
    print("Saved:", PRED_FILE)


if __name__ == "__main__":
    main()
