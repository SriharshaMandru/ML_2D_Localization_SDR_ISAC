import os
import numpy as np
import pandas as pd

from scripts.utils.plot_utils import save_true_pred_plot, save_cdf


def main():
    os.makedirs("results/plots", exist_ok=True)

    if os.path.exists("results/predictions/angle_predictions.csv"):
        a = pd.read_csv("results/predictions/angle_predictions.csv")
        save_true_pred_plot(
            a["true_angle_deg"],
            a["pred_angle_deg"],
            "results/plots/true_vs_predicted_angle.png",
            "True Angle (deg)",
            "Predicted Angle (deg)",
            "True vs Predicted AoA",
        )

    if os.path.exists("results/predictions/distance_predictions.csv"):
        d = pd.read_csv("results/predictions/distance_predictions.csv")
        save_true_pred_plot(
            d["true_distance_m"],
            d["pred_distance_m"],
            "results/plots/true_vs_predicted_distance.png",
            "True Distance (m)",
            "Predicted Distance (m)",
            "True vs Predicted Distance",
        )

    if os.path.exists("results/predictions/xy_predictions.csv"):
        xy = pd.read_csv("results/predictions/xy_predictions.csv")
        err = np.sqrt((xy["pred_x"] - xy["true_x"]) ** 2 + (xy["pred_y"] - xy["true_y"]) ** 2)
        save_cdf(err, "results/plots/localization_error_cdf.png", "Localization Error CDF")

    print("Plots saved in results/plots")


if __name__ == "__main__":
    main()
