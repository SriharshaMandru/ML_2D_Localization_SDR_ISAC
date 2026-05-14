import os
import glob
import numpy as np
import pandas as pd

OUT_FILE = "results/predictions/model_comparison.csv"


def summarize_prediction_file(path):
    df = pd.read_csv(path)

    if not {"true_x", "true_y", "pred_x", "pred_y"}.issubset(df.columns):
        return None

    err = np.sqrt((df["pred_x"] - df["true_x"]) ** 2 + (df["pred_y"] - df["true_y"]) ** 2)

    return {
        "model": os.path.basename(path).replace("_xy_predictions.csv", "").replace(".csv", ""),
        "samples": len(df),
        "mean_error_m": np.mean(err),
        "median_error_m": np.median(err),
        "p90_error_m": np.percentile(err, 90),
        "max_error_m": np.max(err),
    }


def main():
    rows = []

    for path in glob.glob("results/predictions/*xy_predictions.csv"):
        s = summarize_prediction_file(path)
        if s:
            rows.append(s)

    if not rows:
        print("No XY prediction files found.")
        return

    out = pd.DataFrame(rows).sort_values("mean_error_m")
    os.makedirs("results/predictions", exist_ok=True)
    out.to_csv(OUT_FILE, index=False)

    print("Saved:", OUT_FILE)
    print(out)


if __name__ == "__main__":
    main()
