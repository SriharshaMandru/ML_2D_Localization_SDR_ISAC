import numpy as np
import pandas as pd

PRED_FILE = "results/predictions/angle_predictions.csv"


def main():
    p = pd.read_csv(PRED_FILE)
    e = p["pred_angle_deg"] - p["true_angle_deg"]

    print("Angle MAE:", np.mean(np.abs(e)))
    print("Angle RMSE:", np.sqrt(np.mean(e ** 2)))
    print("Angle bias:", np.mean(e))


if __name__ == "__main__":
    main()
