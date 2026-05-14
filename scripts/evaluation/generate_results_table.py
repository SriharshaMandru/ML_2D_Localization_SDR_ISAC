import os
import pandas as pd


def copy_if_exists(src, dst):
    if os.path.exists(src):
        pd.read_csv(src).to_csv(dst, index=False)
        print("Saved:", dst)


def main():
    os.makedirs("results/tables", exist_ok=True)

    copy_if_exists("results/predictions/model_comparison.csv", "results/tables/final_model_comparison_table.csv")
    copy_if_exists("results/predictions/xy_predictions.csv", "results/tables/final_xy_predictions_table.csv")

    print("Result table generation completed.")


if __name__ == "__main__":
    main()
