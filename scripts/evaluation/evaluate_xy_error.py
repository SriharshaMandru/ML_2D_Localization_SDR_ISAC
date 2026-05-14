import numpy as np
import pandas as pd

PRED_FILE = "results/predictions/xy_predictions.csv"


def main():
    p = pd.read_csv(PRED_FILE)

    e = np.sqrt((p["pred_x"] - p["true_x"]) ** 2 + (p["pred_y"] - p["true_y"]) ** 2)

    print("Localization mean error:", np.mean(e))
    print("Localization median error:", np.median(e))
    print("Localization P90 error:", np.percentile(e, 90))


if __name__ == "__main__":
    main()
