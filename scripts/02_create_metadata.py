#!/usr/bin/env python3
import os
import glob
import re
import pandas as pd

RX_ROOT = "data/raw_iq/rx"
TX_ROOT = "data/raw_iq/tx"
OUT_FILE = "data/metadata/measurement_log.csv"

# Folder angle correction:
# folder says negative, but real measured data is positive.
ANGLE_CORRECTION = {
    -60: 60,
    -45: 45,
    -30: 30,
    -15: 15,
      0: 0,
     15: 15,
     30: 30,
     45: 45,
     60: 60,
}


def normalize_pair(pair):
    pair = str(pair).strip()

    mapping = {
        "(1,3)": ["(1,3)", "pair13", "13", "1,3"],
        "(1,4)": ["(1,4)", "pair14", "14", "1,4"],
        "(2,3)": ["(2,3)", "pair23", "23", "2,3"],
        "(2,4)": ["(2,4)", "pair24", "24", "2,4"],
        "pair13": ["pair13", "(1,3)", "13", "1,3"],
        "pair14": ["pair14", "(1,4)", "14", "1,4"],
        "pair23": ["pair23", "(2,3)", "23", "2,3"],
        "pair24": ["pair24", "(2,4)", "24", "2,4"],
    }

    return mapping.get(pair, [pair])


def parse_condition_from_rx_dat_path(path):
    norm = os.path.normpath(path)
    parts = norm.split(os.sep)

    if "rx" not in parts:
        raise ValueError(f"Path does not contain rx folder: {path}")

    idx = parts.index("rx")

    freq_str = parts[idx + 1]       # 850MHz
    dist_str = parts[idx + 2]       # 1m / 1M
    pair = parts[idx + 3]           # (1,3) / pair13
    angle_str = parts[idx + 4]      # -60 / 60 / 60deg
    file_name = parts[idx + 5]      # rep1_rx.dat

    freq_match = re.search(r"\d+(\.\d+)?", freq_str)
    dist_match = re.search(r"\d+(\.\d+)?", dist_str)
    angle_match = re.search(r"-?\d+(\.\d+)?", angle_str)

    if not freq_match:
        raise ValueError(f"Frequency not found in folder: {freq_str}")

    if not dist_match:
        raise ValueError(f"Distance not found in folder: {dist_str}")

    if not angle_match:
        raise ValueError(f"Angle not found in folder: {angle_str}")

    folder_angle_deg = int(float(angle_match.group(0)))

    if folder_angle_deg not in ANGLE_CORRECTION:
        raise ValueError(
            f"Angle {folder_angle_deg} not present in ANGLE_CORRECTION map"
        )

    true_angle_deg = ANGLE_CORRECTION[folder_angle_deg]

    rep_match = re.search(r"rep(\d+)", file_name, flags=re.IGNORECASE)
    rep = "rep" + rep_match.group(1).zfill(2) if rep_match else "rep00"

    return {
        "rx_file_name": file_name,
        "rx_file_path": path,
        "frequency_mhz": float(freq_match.group(0)),
        "distance_m": float(dist_match.group(0)),
        "pair": pair,
        "folder_angle_deg": folder_angle_deg,
        "angle_deg": true_angle_deg,
        "rep": rep,
    }


def find_tx_for_condition(meta):
    freq_number = int(meta["frequency_mhz"])
    distance_number = int(meta["distance_m"])

    # Use folder angle for TX path matching, because TX folder follows your actual folder name.
    angle_number = int(meta["folder_angle_deg"])

    freq_variants = [
        f"{freq_number}MHz",
        f"{freq_number}mhz",
        str(freq_number),
    ]

    distance_variants = [
        f"{distance_number}m",
        f"{distance_number}M",
        str(distance_number),
    ]

    pair_variants = normalize_pair(meta["pair"])

    angle_variants = [
        str(angle_number),
        f"{angle_number}deg",
        f"{angle_number}Deg",
        f"{angle_number}DEG",
    ]

    tx_name_variants = [
        "IQ_rep_Tx.dat",
        "IQ_rep_TX.dat",
        "iq_rep_tx.dat",
        "IQ_TX.dat",
        "IQ_Tx.dat",
        "tx_ref.dat",
        "tx.dat",
        "TX.dat",
    ]

    for freq_folder in freq_variants:
        for dist_folder in distance_variants:
            for pair_folder in pair_variants:
                for angle_folder in angle_variants:

                    tx_condition_dir = os.path.join(
                        TX_ROOT,
                        freq_folder,
                        dist_folder,
                        pair_folder,
                        angle_folder,
                    )

                    for tx_name in tx_name_variants:
                        tx_path = os.path.join(tx_condition_dir, tx_name)
                        if os.path.isfile(tx_path):
                            return tx_path, True

                    dat_files = glob.glob(os.path.join(tx_condition_dir, "*.dat"))

                    if dat_files:
                        return sorted(dat_files)[0], True

    all_tx_files = glob.glob(os.path.join(TX_ROOT, "**", "*.dat"), recursive=True)

    wanted_freq = str(freq_number)
    wanted_dist = str(distance_number)
    wanted_angle = str(angle_number)

    pair_tokens = []
    for p in pair_variants:
        pair_tokens.append(
            p.replace("(", "").replace(")", "").replace(",", "")
        )

    candidates = []

    for tx_file in all_tx_files:
        norm = os.path.normpath(tx_file)
        full = "/".join(norm.split(os.sep))

        has_freq = wanted_freq in full
        has_dist = any(
            x in full
            for x in [
                f"{wanted_dist}m",
                f"{wanted_dist}M",
                f"/{wanted_dist}/",
            ]
        )
        has_angle = any(
            x in full
            for x in [
                f"/{wanted_angle}/",
                f"{wanted_angle}deg",
                f"{wanted_angle}Deg",
            ]
        )

        clean_full = full.replace("(", "").replace(")", "").replace(",", "")
        has_pair = any(p in clean_full for p in pair_tokens)

        if has_freq and has_dist and has_angle and has_pair:
            candidates.append(tx_file)

    if candidates:
        return sorted(candidates)[0], True

    expected_tx_path = os.path.join(
        TX_ROOT,
        f"{freq_number}MHz",
        f"{distance_number}m",
        meta["pair"],
        str(angle_number),
        "IQ_rep_Tx.dat",
    )

    return expected_tx_path, False


def main():
    os.makedirs("data/metadata", exist_ok=True)

    rx_files = glob.glob(
        os.path.join(RX_ROOT, "**", "*.dat"),
        recursive=True
    )

    rx_files = sorted(set(rx_files))

    rows = []

    for rx_path in rx_files:
        try:
            meta = parse_condition_from_rx_dat_path(rx_path)
            tx_path, tx_available = find_tx_for_condition(meta)

            rows.append({
                "rx_file_name": meta["rx_file_name"],
                "rx_file_path": meta["rx_file_path"],
                "tx_file_path": tx_path,
                "tx_available": tx_available,
                "frequency_mhz": meta["frequency_mhz"],
                "distance_m": meta["distance_m"],
                "pair": meta["pair"],

                # Original folder label
                "folder_angle_deg": meta["folder_angle_deg"],

                # Corrected true measured label used for ML
                "angle_deg": meta["angle_deg"],

                "rep": meta["rep"],
            })

        except Exception as e:
            print("Skipping:", rx_path)
            print("Reason:", e)

    df = pd.DataFrame(rows)

    if df.empty:
        print("No RX .dat files found.")
        return

    df = df.sort_values(
        [
            "frequency_mhz",
            "distance_m",
            "pair",
            "folder_angle_deg",
            "rep",
        ]
    ).reset_index(drop=True)

    df.to_csv(OUT_FILE, index=False)

    print("Metadata saved:", OUT_FILE)
    print("RX files:", len(df))
    print("TX matched:", int(df["tx_available"].sum()))
    print("TX missing:", int((~df["tx_available"]).sum()))

    print("\nFolder angle distribution:")
    print(df["folder_angle_deg"].value_counts().sort_index())

    print("\nCorrected ML angle distribution:")
    print(df["angle_deg"].value_counts().sort_index())

    print("\nDistance distribution:")
    print(df["distance_m"].value_counts().sort_index())

    print("\nPair distribution:")
    print(df["pair"].value_counts().sort_index())

    print("\nPreview:")
    print(df.head(20))

    missing = df[df["tx_available"] == False]
    if len(missing) > 0:
        print("\nTX missing preview:")
        print(
            missing[
                [
                    "frequency_mhz",
                    "distance_m",
                    "pair",
                    "folder_angle_deg",
                    "angle_deg",
                    "rep",
                    "tx_file_path",
                ]
            ].head(30)
        )


if __name__ == "__main__":
    main()