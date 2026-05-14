import os
import re
import numpy as np
import pandas as pd


def parse_condition_from_rx_path(path):
    """
    Expected RX:
    data/raw_iq/rx/850MHz/1M/(1,3)/-60/rep01.csv
    """
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)

    if "rx" not in parts:
        raise ValueError(f"Path does not contain rx folder: {path}")

    idx = parts.index("rx")

    freq_str = parts[idx + 1]
    dist_str = parts[idx + 2]
    pair = parts[idx + 3]
    angle_str = parts[idx + 4]
    fname = parts[idx + 5]

    rep_match = re.search(r"rep(\d+)", fname, flags=re.IGNORECASE)
    rep = rep_match.group(1).zfill(2) if rep_match else None

    return {
        "frequency_mhz": float(freq_str.replace("MHz", "")),
        "distance_m": float(dist_str.replace("M", "")),
        "pair": pair,
        "angle_deg": float(angle_str),
        "rep": rep,
        "rx_file_name": fname,
    }


def read_rx_csv(path, max_samples=None):
    df = pd.read_csv(path)

    required = ["I1", "Q1", "I2", "Q2"]

    if not all(c in df.columns for c in required):
        raise ValueError(f"{path} must contain I1,Q1,I2,Q2")

    if max_samples is not None:
        df = df.iloc[:int(max_samples)]

    x1 = df["I1"].to_numpy(float) + 1j * df["Q1"].to_numpy(float)
    x2 = df["I2"].to_numpy(float) + 1j * df["Q2"].to_numpy(float)

    return x1, x2


def read_tx_csv(path, max_samples=None):
    df = pd.read_csv(path)

    if {"I", "Q"}.issubset(df.columns):
        i_col = "I"
        q_col = "Q"
    elif {"I_tx", "Q_tx"}.issubset(df.columns):
        i_col = "I_tx"
        q_col = "Q_tx"
    else:
        raise ValueError(f"{path} must contain I,Q or I_tx,Q_tx")

    if max_samples is not None:
        df = df.iloc[:int(max_samples)]

    return df[i_col].to_numpy(float) + 1j * df[q_col].to_numpy(float)


def remove_dc(x):
    return x - np.mean(x)


def normalize_power(x):
    p = np.mean(np.abs(x) ** 2)
    if p <= 0:
        return x
    return x / np.sqrt(p)


def clean_rx(x1, x2, remove_dc_flag=True, normalize_power_flag=True, clip_extreme=True):
    mask = (
        np.isfinite(x1.real)
        & np.isfinite(x1.imag)
        & np.isfinite(x2.real)
        & np.isfinite(x2.imag)
    )

    x1 = x1[mask]
    x2 = x2[mask]

    if remove_dc_flag:
        x1 = remove_dc(x1)
        x2 = remove_dc(x2)

    if clip_extreme:
        a1 = np.abs(x1)
        a2 = np.abs(x2)
        limit1 = np.median(a1) + 6 * np.std(a1)
        limit2 = np.median(a2) + 6 * np.std(a2)
        mask = (a1 <= limit1) & (a2 <= limit2)
        x1 = x1[mask]
        x2 = x2[mask]

    if normalize_power_flag:
        x1 = normalize_power(x1)
        x2 = normalize_power(x2)

    return x1, x2


def clean_tx(x, remove_dc_flag=True, normalize_power_flag=True, clip_extreme=True):
    x = x[np.isfinite(x.real) & np.isfinite(x.imag)]

    if remove_dc_flag:
        x = remove_dc(x)

    if clip_extreme:
        a = np.abs(x)
        limit = np.median(a) + 6 * np.std(a)
        x = x[a <= limit]

    if normalize_power_flag:
        x = normalize_power(x)

    return x


def save_rx_csv(path, x1, x2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = min(len(x1), len(x2))

    pd.DataFrame(
        {
            "I1": x1[:n].real,
            "Q1": x1[:n].imag,
            "I2": x2[:n].real,
            "Q2": x2[:n].imag,
        }
    ).to_csv(path, index=False)


def save_tx_csv(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    pd.DataFrame(
        {
            "I": x.real,
            "Q": x.imag,
        }
    ).to_csv(path, index=False)


def safe_pair_name(pair):
    return "pair" + pair.replace("(", "").replace(")", "").replace(",", "")
