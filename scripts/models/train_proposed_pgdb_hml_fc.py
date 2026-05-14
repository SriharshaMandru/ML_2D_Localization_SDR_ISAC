import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.utils.geometry_utils import angle_distance_to_xy, feedback_correct_xy
from scripts.utils.model_utils import numeric_feature_columns
from scripts.utils.sign_correct_aoa import apply_sign_correction_batch

DATASET_FILE = "data/processed/final_dataset_split.csv"
ANGLE_MODEL_FILE = "models_saved/angle_branch_extratrees.pkl"
DIST_MODEL_FILE = "models_saved/distance_branch_xgboost.pkl"
HYBRID_MODEL_FILE = "models_saved/proposed_pgdb_hml_fc.pkl"
PRED_FILE = "results/predictions/xy_predictions.csv"
TABLE_FILE = "results/tables/localization_error_table.csv"


def localization_error(true_xy, pred_xy):
    return np.sqrt((pred_xy[:, 0] - true_xy[:, 0]) ** 2 + (pred_xy[:, 1] - true_xy[:, 1]) ** 2)


def print_metrics(name, true_xy, pred_xy):
    err = localization_error(true_xy, pred_xy)
    print(f"\n{name}")
    print("Mean error :", np.mean(err))
    print("Median error:", np.median(err))
    print("P90 error  :", np.percentile(err, 90))
    print("RMSE XY    :", mean_squared_error(true_xy, pred_xy) ** 0.5)
    print("MAE XY     :", mean_absolute_error(true_xy, pred_xy))
    print("R2 X       :", r2_score(true_xy[:, 0], pred_xy[:, 0]))
    print("R2 Y       :", r2_score(true_xy[:, 1], pred_xy[:, 1]))
    return err


def main():
    df = pd.read_csv(DATASET_FILE)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    angle_pack = joblib.load(ANGLE_MODEL_FILE)
    dist_pack = joblib.load(DIST_MODEL_FILE)

    angle_model = angle_pack["model"]
    distance_model = dist_pack["model"]
    angle_features = angle_pack["feature_cols"]
    distance_features = dist_pack["feature_cols"]

    def predict_angle_distance(part):
        # Raw magnitude from the angle model (trained on positive angles only)
        pred_angle_raw = angle_model.predict(part[angle_features])
        # Apply physical sign from TX-corrected inter-element phase differences
        pred_angle = apply_sign_correction_batch(pred_angle_raw, part)
        pred_distance = distance_model.predict(part[distance_features])
        pred_xy = np.array([angle_distance_to_xy(a, d) for a, d in zip(pred_angle, pred_distance)])
        return pred_angle, pred_distance, pred_xy

    train_angle, train_distance, train_geom_xy = predict_angle_distance(train_df)
    val_angle, val_distance, val_geom_xy = predict_angle_distance(val_df)
    test_angle, test_distance, test_geom_xy = predict_angle_distance(test_df)

    train_true_xy = train_df[["target_x", "target_y"]].to_numpy()
    val_true_xy = val_df[["target_x", "target_y"]].to_numpy()
    test_true_xy = test_df[["target_x", "target_y"]].to_numpy()

    drop_cols = [
        "split",
        "angle_deg",
        "distance_m",
        "target_angle_deg",
        "target_distance_m",
        "target_x",
        "target_y",
    ]

    hybrid_features = numeric_feature_columns(df, drop_cols)

    residual_target = train_true_xy - train_geom_xy

    residual_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            min_samples_leaf=2,
        )
    )

    residual_model.fit(train_df[hybrid_features], residual_target)

    def final_xy(part, pred_angle, pred_distance, geom_xy):
        residual = residual_model.predict(part[hybrid_features])
        corrected_xy = geom_xy + residual
        feedback_xy = np.array(
            [
                feedback_correct_xy(xy, a, d, iterations=3, lr=0.25)
                for xy, a, d in zip(corrected_xy, pred_angle, pred_distance)
            ]
        )
        return corrected_xy, feedback_xy

    val_residual_xy, val_final_xy = final_xy(val_df, val_angle, val_distance, val_geom_xy)
    test_residual_xy, test_final_xy = final_xy(test_df, test_angle, test_distance, test_geom_xy)

    print_metrics("VALIDATION FINAL LOCALIZATION", val_true_xy, val_final_xy)
    test_err = print_metrics("TEST FINAL LOCALIZATION", test_true_xy, test_final_xy)

    os.makedirs("models_saved", exist_ok=True)
    joblib.dump(
        {
            "angle_model_file": ANGLE_MODEL_FILE,
            "distance_model_file": DIST_MODEL_FILE,
            "residual_model": residual_model,
            "angle_features": angle_features,
            "distance_features": distance_features,
            "hybrid_features": hybrid_features,
            "description": "PGDB-HML-FC: AoA + RSSI distance + residual ML + feedback correction",
        },
        HYBRID_MODEL_FILE,
    )

    os.makedirs("results/predictions", exist_ok=True)

    pred_df = test_df[["frequency_mhz", "distance_m", "angle_deg", "rep", "window_id"]].copy()
    pred_df["true_x"] = test_true_xy[:, 0]
    pred_df["true_y"] = test_true_xy[:, 1]
    pred_df["pred_angle_deg"] = test_angle
    pred_df["pred_distance_m"] = test_distance
    pred_df["geom_x"] = test_geom_xy[:, 0]
    pred_df["geom_y"] = test_geom_xy[:, 1]
    pred_df["residual_x"] = test_residual_xy[:, 0]
    pred_df["residual_y"] = test_residual_xy[:, 1]
    pred_df["pred_x"] = test_final_xy[:, 0]
    pred_df["pred_y"] = test_final_xy[:, 1]
    pred_df["localization_error_m"] = test_err
    pred_df.to_csv(PRED_FILE, index=False)

    os.makedirs("results/tables", exist_ok=True)

    summary = pred_df.groupby(["frequency_mhz", "distance_m"]).agg(
        samples=("localization_error_m", "count"),
        mean_error_m=("localization_error_m", "mean"),
        median_error_m=("localization_error_m", "median"),
        p90_error_m=("localization_error_m", lambda x: np.percentile(x, 90)),
        max_error_m=("localization_error_m", "max"),
    ).reset_index()

    summary.to_csv(TABLE_FILE, index=False)

    print("\nSaved:", HYBRID_MODEL_FILE)
    print("Saved:", PRED_FILE)
    print("Saved:", TABLE_FILE)


if __name__ == "__main__":
    main()
