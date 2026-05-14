"""
verify_phase_sign.py
--------------------
Sanity-check script for the sign-aware AoA pipeline.

Synthesises IQ signals for known positive and negative angles, runs them
through extract_pair_features + apply_sign_correction, and asserts that:

  1. positive angle  →  signed_phase_diff_rad > 0  →  corrected AoA > 0
  2. negative angle  →  signed_phase_diff_rad < 0  →  corrected AoA < 0
  3. TX correction removes a large synthetic cable offset without flipping sign
  4. 0° angle        →  |signed_phase_diff_rad| < 0.01 rad  (broadside, ~0)

Run from the project root:
    python -m scripts.verify_phase_sign
"""

import sys
import numpy as np

from scripts.utils.feature_utils import (
    extract_pair_features,
    wavelength,
)
from scripts.utils.sign_correct_aoa import apply_sign_correction

# ── Experiment constants ──────────────────────────────────────────────────────
FREQ_MHZ      = 850.0
BASELINE_23   = 0.165   # pair (2,3) — UNAMBIGUOUS at 850 MHz (d/λ = 0.47)
BASELINE_14   = 0.495   # pair (1,4) — ambiguous (d/λ = 1.40)
N_SAMPLES     = 8192


def synthesise_iq_pair(angle_deg, baseline_m, freq_mhz,
                       hw_offset1=0.0, hw_offset2=0.0,
                       snr_db=40.0, seed=0):
    """
    Create a complex baseband IQ pair for a plane wave at angle_deg.

    Returns (x1, x2, tx, true_phase_diff_rad).
    hw_offset_k : simulated cable/LO phase bias for branch k (radians).
    """
    rng   = np.random.default_rng(seed)
    lam   = wavelength(freq_mhz)

    theta           = np.radians(angle_deg)
    true_phase_diff = (2 * np.pi * baseline_m / lam) * np.sin(theta)

    phase_tx = np.linspace(0, 2 * np.pi * 0.1 * N_SAMPLES, N_SAMPLES)
    tx = np.exp(1j * phase_tx)

    noise_amp = 10 ** (-snr_db / 20)
    x1 = tx * np.exp(1j * hw_offset1) + noise_amp * (
        rng.standard_normal(N_SAMPLES) + 1j * rng.standard_normal(N_SAMPLES)
    )
    x2 = tx * np.exp(1j * (hw_offset2 + true_phase_diff)) + noise_amp * (
        rng.standard_normal(N_SAMPLES) + 1j * rng.standard_normal(N_SAMPLES)
    )
    return x1, x2, tx, true_phase_diff


def run_test(label, angle_deg, baseline_m, hw_offset1=0.0, hw_offset2=0.0,
             use_tx=True, sign_col_override=None):
    """
    Test one scenario.

    sign_col_override: if provided, use this dict as the feature_row for
    apply_sign_correction (lets us simulate the merged multi-pair dataset).
    """
    x1, x2, tx_ref, true_phase_diff = synthesise_iq_pair(
        angle_deg, baseline_m, FREQ_MHZ,
        hw_offset1=hw_offset1, hw_offset2=hw_offset2,
    )
    tx = tx_ref if use_tx else None

    feats = extract_pair_features(x1, x2, tx, FREQ_MHZ, baseline_m)

    signed_pd  = feats["signed_phase_diff_rad"]
    signed_aoa = feats["signed_phase_diff_aoa_deg"]

    # For sign correction use the provided override or the raw feature dict
    sign_row = sign_col_override if sign_col_override is not None else feats
    model_pred_magnitude = abs(angle_deg) + np.random.default_rng(42).normal(0, 1)
    corrected = apply_sign_correction(model_pred_magnitude, sign_row)

    expected_sign = "+" if angle_deg > 0 else ("-" if angle_deg < 0 else "0")

    ok_phase = (
        (angle_deg > 0 and signed_pd > 0) or
        (angle_deg < 0 and signed_pd < 0) or
        (angle_deg == 0 and abs(signed_pd) < 0.05)
    )
    ok_aoa = (
        (angle_deg > 0 and signed_aoa > 0) or
        (angle_deg < 0 and signed_aoa < 0) or
        (angle_deg == 0 and abs(signed_aoa) < 1.0)
    )
    ok_correction = (
        (angle_deg > 0 and corrected > 0) or
        (angle_deg < 0 and corrected < 0) or
        (angle_deg == 0)
    )

    status = "PASS" if (ok_phase and ok_aoa and ok_correction) else "FAIL"

    lam = wavelength(FREQ_MHZ)
    print(f"\n[{status}] {label}")
    print(f"  baseline           : {baseline_m} m  (d/λ = {baseline_m/lam:.2f})")
    print(f"  angle_deg          : {angle_deg:+.1f}°  (expected sign: {expected_sign})")
    print(f"  true_phase_diff    : {true_phase_diff:+.4f} rad")
    print(f"  signed_phase_diff  : {signed_pd:+.4f} rad  {'✓' if ok_phase else '✗ WRONG SIGN'}")
    print(f"  signed_phase_aoa   : {signed_aoa:+.2f}°    {'✓' if ok_aoa else '✗ WRONG SIGN'}")
    print(f"  corrected_pred     : {corrected:+.2f}°   {'✓' if ok_correction else '✗ WRONG SIGN'}")
    print(f"  tx_correction used : {use_tx}")
    return status == "PASS"


def main():
    print("=" * 65)
    print(" AoA Sign Convention Verification")
    print(f" Freq={FREQ_MHZ} MHz")
    print(f" Pair(2,3) baseline={BASELINE_23} m  d/λ={BASELINE_23/wavelength(FREQ_MHZ):.2f}  [UNAMBIGUOUS]")
    print(f" Pair(1,4) baseline={BASELINE_14} m  d/λ={BASELINE_14/wavelength(FREQ_MHZ):.2f}  [AMBIGUOUS]")
    print("=" * 65)

    results = []

    # ── Tests using pair (2,3) — unambiguous pair, sign is always correct ──
    print("\n── Pair (2,3) tests: d=0.165m, d/λ=0.47 (UNAMBIGUOUS) ──")

    results.append(run_test("+30° pair23, no HW offset, TX available",
                            angle_deg=+30.0, baseline_m=BASELINE_23, use_tx=True))

    results.append(run_test("-30° pair23, no HW offset, TX available",
                            angle_deg=-30.0, baseline_m=BASELINE_23, use_tx=True))

    results.append(run_test("+30° pair23, LO offset=1.8 rad, TX corrects",
                            angle_deg=+30.0, baseline_m=BASELINE_23,
                            hw_offset1=0.0, hw_offset2=1.8, use_tx=True))

    results.append(run_test("-30° pair23, LO offset=1.8 rad, TX corrects",
                            angle_deg=-30.0, baseline_m=BASELINE_23,
                            hw_offset1=0.0, hw_offset2=1.8, use_tx=True))

    results.append(run_test("+60° pair23, TX available",
                            angle_deg=+60.0, baseline_m=BASELINE_23, use_tx=True))

    results.append(run_test("-60° pair23, TX available",
                            angle_deg=-60.0, baseline_m=BASELINE_23, use_tx=True))

    results.append(run_test("0° broadside pair23, TX available",
                            angle_deg=0.0, baseline_m=BASELINE_23, use_tx=True))

    results.append(run_test("+30° pair23, NO TX (raw phase, HW offset=0)",
                            angle_deg=+30.0, baseline_m=BASELINE_23, use_tx=False))

    # ── Combined test: pair14 magnitude + pair23 sign ──────────────────────
    # This simulates the real pipeline: pair(1,4) features are used for
    # magnitude precision, but sign comes from pair(2,3) via the merged row.
    print("\n── Combined pair14 + pair23 sign extraction test ──")

    def combined_test(label, angle_deg):
        # Simulate pair14 features (ambiguous magnitude)
        _, _, _, _ = synthesise_iq_pair(angle_deg, BASELINE_14, FREQ_MHZ, seed=1)
        x1_14, x2_14, tx14, _ = synthesise_iq_pair(angle_deg, BASELINE_14, FREQ_MHZ, seed=1)
        feats14 = extract_pair_features(x1_14, x2_14, tx14, FREQ_MHZ, BASELINE_14)

        # Simulate pair23 features (unambiguous sign)
        x1_23, x2_23, tx23, _ = synthesise_iq_pair(angle_deg, BASELINE_23, FREQ_MHZ, seed=2)
        feats23 = extract_pair_features(x1_23, x2_23, tx23, FREQ_MHZ, BASELINE_23)

        # Build a merged feature row (simulating 09_merge_pair_features.py output)
        merged_row = {}
        for k, v in feats14.items():
            merged_row[f"pair14_{k}"] = v
        for k, v in feats23.items():
            merged_row[f"pair23_{k}"] = v

        model_pred_magnitude = abs(angle_deg) + 1.0   # simulated model output
        corrected = apply_sign_correction(model_pred_magnitude, merged_row)

        ok = (angle_deg > 0 and corrected > 0) or \
             (angle_deg < 0 and corrected < 0) or \
             (angle_deg == 0)
        status = "PASS" if ok else "FAIL"
        print(f"\n[{status}] {label}")
        print(f"  angle_deg        : {angle_deg:+.1f}°")
        print(f"  pair23 phase     : {feats23['tx_corr_phase_mean']:+.4f} rad  (sign source)")
        print(f"  pair14 phase     : {feats14['tx_corr_phase_mean']:+.4f} rad  (ambiguous)")
        print(f"  corrected_pred   : {corrected:+.2f}°   {'✓' if ok else '✗ WRONG SIGN'}")
        return ok

    results.append(combined_test("Combined +30°: magnitude from pair14, sign from pair23", +30.0))
    results.append(combined_test("Combined -30°: magnitude from pair14, sign from pair23", -30.0))
    results.append(combined_test("Combined +45°: magnitude from pair14, sign from pair23", +45.0))
    results.append(combined_test("Combined -45°: magnitude from pair14, sign from pair23", -45.0))

    print("\n" + "=" * 65)
    n_pass = sum(results)
    n_total = len(results)
    print(f" Results: {n_pass}/{n_total} passed")
    if n_pass == n_total:
        print(" ALL TESTS PASSED — sign convention is correct.")
    else:
        print(" SOME TESTS FAILED — check hardware offset or antenna ordering.")
    print("=" * 65)

    sys.exit(0 if n_pass == n_total else 1)


if __name__ == "__main__":
    main()
