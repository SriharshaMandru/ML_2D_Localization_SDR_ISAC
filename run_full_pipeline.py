import os
import subprocess
import sys

STEPS = [
    ("Check IQ files", "scripts/00_check_iq_files.py"),
    ("Create metadata", "scripts/02_create_metadata.py"),
    ("Preprocess TX/RX IQ", "scripts/03_preprocess_iq.py"),
    ("Window RX IQ", "scripts/04_window_iq.py"),
    ("Extract pair (1,4) features", "scripts/05_extract_features_pair14.py"),
    ("Extract pair (2,3) features", "scripts/06_extract_features_pair23.py"),
    ("Extract pair (1,3) features", "scripts/07_extract_features_pair13.py"),
    ("Extract pair (2,4) features", "scripts/08_extract_features_pair24.py"),
    ("Merge all pair features", "scripts/09_merge_pair_features.py"),
    ("Create final dataset", "scripts/10_create_final_dataset.py"),
    ("Split dataset", "scripts/11_split_dataset.py"),
    ("Train AoA model", "scripts/models/train_angle_branch_extratrees.py"),
    ("Train RSSI distance model", "scripts/models/train_distance_branch_xgboost.py"),
    ("Train hybrid feedback model", "scripts/models/train_proposed_pgdb_hml_fc.py"),
    ("Train Random Forest comparison", "scripts/models/train_random_forest.py"),
    ("Train Extra Trees comparison", "scripts/models/train_extra_trees.py"),
    ("Train XGBoost comparison", "scripts/models/train_xgboost.py"),
    ("Train MLP comparison", "scripts/models/train_mlp.py"),
    ("Train Direct XY comparison", "scripts/models/train_direct_xy_regression.py"),
    ("Evaluate AoA", "scripts/evaluation/evaluate_angle_error.py"),
    ("Evaluate Distance", "scripts/evaluation/evaluate_distance_error.py"),
    ("Evaluate XY", "scripts/evaluation/evaluate_xy_error.py"),
    ("Compare models", "scripts/evaluation/compare_models.py"),
    ("Generate results tables", "scripts/evaluation/generate_results_table.py"),
    ("Make plots", "scripts/evaluation/make_plots.py"),
]


def run_step(name, script):
    print("\n" + "=" * 80)
    print("RUNNING:", name)
    print("SCRIPT :", script)
    print("=" * 80)

    if not os.path.exists(script):
        print("Missing script:", script)
        sys.exit(1)

    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print("FAILED:", name)
        sys.exit(result.returncode)


def main():
    required_dirs = [
        "data/metadata",
        "data/processed/preprocessed/tx",
        "data/processed/preprocessed/rx",
        "data/processed/windows",
        "data/processed/features",
        "models_saved",
        "results/predictions",
        "results/tables",
        "results/plots",
    ]

    for d in required_dirs:
        os.makedirs(d, exist_ok=True)

    for name, script in STEPS:
        run_step(name, script)

    print("\nFULL PIPELINE COMPLETED")
    print("Final hybrid model: models_saved/proposed_pgdb_hml_fc.pkl")


if __name__ == "__main__":
    main()
