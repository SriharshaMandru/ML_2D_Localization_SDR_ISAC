import numpy as np
import pandas as pd

PRED_FILE = "results/predictions/distance_predictions.csv"


def main():
    p = pd.read_csv(PRED_FILE)
    e = p["pred_distance_m"] - p["true_distance_m"]

    print("Distance MAE:", np.mean(np.abs(e)))
    print("Distance RMSE:", np.sqrt(np.mean(e ** 2)))
    print("Distance bias:", np.mean(e))


if __name__ == "__main__":
    main()
