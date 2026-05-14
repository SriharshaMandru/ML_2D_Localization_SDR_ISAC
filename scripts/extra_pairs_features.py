#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

META_FILE = "data/metadata/measurement_log.csv"
OUT_FILE = "data/processed/pairwise_features.csv"

WINDOW_SIZE = 8192
STEP_SIZE = 4096

# Equal windows per RX .dat file
MAX_WINDOWS_PER_FILE = 250

C = 3e8


def read_dat_iq_file(path):
    raw = np.fromfile(path, dtype=np.complex64)

    if len(raw) < 2:
        raise ValueError("DAT file too small")

    if len(raw) % 2 != 0:
        raw = raw[:-1]

    # Assumed DAT format:
    # rx1_sample0, rx2_sample0, rx1_sample1, rx2_sample1, ...
    x1 = raw[0::2]
    x2 = raw[1::2]

    return x1, x2


def preprocess_signal(x):
    # DC offset removal
    x = x - np.mean(x)

    # Amplitude normalization
    max_val = np.max(np.abs(x))
    if max_val > 0:
        x = x / max_val

    return x


def baseline_for_frequency(fc_mhz):
    wavelength = C / (fc_mhz * 1e6)
    return wavelength / 2.0


def compute_features(x1, x2, fc_mhz, baseline_m):
    x1 = preprocess_signal(x1)
    x2 = preprocess_signal(x2)

    p1 = np.mean(np.abs(x1) ** 2)
    p2 = np.mean(np.abs(x2) ** 2)

    rssi1 = 10 * np.log10(p1 + 1e-12)
    rssi2 = 10 * np.log10(p2 + 1e-12)

    cross = x1 * np.conj(x2)
    phase = np.angle(cross)
    unwrap_phase = np.unwrap(phase)

    R11 = np.mean(x1 * np.conj(x1))
    R12 = np.mean(x1 * np.conj(x2))
    R22 = np.mean(x2 * np.conj(x2))

    cov_phase = np.angle(R12)

    wavelength = C / (fc_mhz * 1e6)
    phase_mean = np.angle(np.mean(np.exp(1j * phase)))

    val = (phase_mean * wavelength) / (2 * np.pi * baseline_m)
    val = np.clip(val, -1, 1)
    aoa_physics_deg = np.rad2deg(np.arcsin(val))

    n_corr = min(2048, len(x1), len(x2))
    corr = np.correlate(x1[:n_corr], x2[:n_corr], mode="full")
    corr_peak = np.max(np.abs(corr)) / (
        np.linalg.norm(x1[:n_corr]) * np.linalg.norm(x2[:n_corr]) + 1e-12
    )

    return {
        "power1": p1,
        "power2": p2,
        "rssi1": rssi1,
        "rssi2": rssi2,
        "rssi_mean": (rssi1 + rssi2) / 2,
        "rssi_diff": rssi1 - rssi2,
        "power_ratio": p1 / (p2 + 1e-12),
        "power_diff": p1 - p2,

        "phase_diff_mean": phase_mean,
        "phase_diff_median": np.median(phase),
        "phase_diff_std": np.std(phase),

        "unwrap_phase_mean": np.mean(unwrap_phase),
        "unwrap_phase_median": np.median(unwrap_phase),
        "unwrap_phase_std": np.std(unwrap_phase),

        "R11_real": np.real(R11),
        "R12_real": np.real(R12),
        "R12_imag": np.imag(R12),
        "R22_real": np.real(R22),
        "cov_phase": cov_phase,

        "corr_peak": corr_peak,
        "aoa_physics_deg": aoa_physics_deg,
    }


def get_balanced_window_starts(n_samples):
    starts = list(range(0, n_samples - WINDOW_SIZE + 1, STEP_SIZE))

    if len(starts) > MAX_WINDOWS_PER_FILE:
        idx = np.linspace(0, len(starts) - 1, MAX_WINDOWS_PER_FILE).astype(int)
        starts = [starts[i] for i in idx]

    return starts


def main():
    if not os.path.exists(META_FILE):
        raise FileNotFoundError(f"Metadata file not found: {META_FILE}")

    df = pd.read_csv(META_FILE)

    if "rx_file_path" in df.columns:
        rx_col = "rx_file_path"
    elif "rx_file" in df.columns:
        rx_col = "rx_file"
    else:
        raise ValueError("Metadata must contain 'rx_file_path' or 'rx_file' column")

    required_cols = [
        rx_col,
        "frequency_mhz",
        "distance_m",
        "angle_deg",
        "pair",
        "rep",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required metadata column: {col}")

    rows = []

    for _, row in df.iterrows():
        path = row[rx_col]
        fc_mhz = float(row["frequency_mhz"])
        baseline_m = baseline_for_frequency(fc_mhz)

        print("Processing:", path)

        if not os.path.exists(path):
            print("Skipped, RX file not found:", path)
            continue

        try:
            x1, x2 = read_dat_iq_file(path)
        except Exception as e:
            print("Read failed:", path, "|", e)
            continue

        n = min(len(x1), len(x2))

        if n < WINDOW_SIZE:
            print("Skipped, not enough samples:", path)
            continue

        x1 = x1[:n]
        x2 = x2[:n]

        starts = get_balanced_window_starts(n)

        win_id = 0

        for start in starts:
            end = start + WINDOW_SIZE

            w1 = x1[start:end]
            w2 = x2[start:end]

            if np.mean(np.abs(w1)) < 1e-6 or np.mean(np.abs(w2)) < 1e-6:
                continue

            try:
                feats = compute_features(
                    w1,
                    w2,
                    fc_mhz,
                    baseline_m
                )
            except Exception as e:
                print("Feature failed:", path, "window", win_id, "|", e)
                continue

            out = {
                "frequency_mhz": fc_mhz,
                "distance_m": float(row["distance_m"]),
                "angle_deg": float(row["angle_deg"]),
                "pair": row["pair"],
                "rep": row["rep"],
                "window_id": win_id,
                "rx_file_path": path,
            }

            if "folder_angle_deg" in df.columns:
                out["folder_angle_deg"] = row["folder_angle_deg"]

            if "tx_file_path" in df.columns:
                out["tx_file_path"] = row["tx_file_path"]

            if "tx_available" in df.columns:
                out["tx_available"] = row["tx_available"]

            out.update(feats)
            rows.append(out)

            win_id += 1

    out_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    out_df.to_csv(OUT_FILE, index=False)

    print("\nSaved:", OUT_FILE)
    print("Shape:", out_df.shape)

    if not out_df.empty:
        print("\nAngle distribution:")
        print(out_df["angle_deg"].value_counts().sort_index())

        print("\nDistance distribution:")
        print(out_df["distance_m"].value_counts().sort_index())

        print("\nPair distribution:")
        print(out_df["pair"].value_counts().sort_index())

        print("\nWindows per file summary:")
        print(out_df.groupby("rx_file_path")["window_id"].count().describe())


if __name__ == "__main__":
    main()