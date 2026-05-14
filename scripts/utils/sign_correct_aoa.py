"""
sign_correct_aoa.py
-------------------
Inference-time AoA sign correction.

The ML model is trained on positive-angle data [0°, 60°] so its raw output is
always a positive magnitude.  The physical sign of the angle-of-arrival is
encoded in the inter-element phase differences — specifically the
`signed_phase_diff_rad` feature that is stored for every antenna pair after
TX-based hardware-offset correction.

Usage
-----
Single prediction (scalar or 1-D array row):

    from scripts.utils.sign_correct_aoa import apply_sign_correction

    pred_aoa = model.predict(feature_row)[0]          # e.g. +28.3°
    corrected = apply_sign_correction(pred_aoa, feature_row_dict)
    # corrected → +28.3° if phase consensus positive, -28.3° if negative

Batch prediction (pandas DataFrame or dict of arrays):

    corrected = apply_sign_correction_batch(pred_angles, feature_df)

Physical basis
--------------
Antenna layout (left → right):
    Ant1 (USRP1-RX2) | Ant2 (USRP1-TxRx) | Ant3 (USRP2-RX2) | Ant4 (USRP2-TxRx)

ULA phase model:
    phase_diff = angle(x_right * conj(x_left)) = (2π·d/λ)·sin(θ)
    → phase_diff > 0  ⟺  θ > 0  (signal from right / positive side)
    → phase_diff < 0  ⟺  θ < 0  (signal from left  / negative side)

The signed_phase_diff_rad stored per pair is already hardware-corrected via the
TX reference signal (cable / LO offsets removed), so its sign reliably reflects
the physical wavefront direction.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Sign detection strategy — ranked by phase unambiguity (d/λ at 850 MHz)
#
#   pair (2,3) : d = 0.165 m  →  d/λ = 0.47  →  UNAMBIGUOUS (d < λ/2)  ✔
#   pair (1,3) : d = 0.330 m  →  d/λ = 0.93  →  ambiguous for |AoA| > 32°
#   pair (2,4) : d = 0.330 m  →  d/λ = 0.93  →  ambiguous for |AoA| > 32°
#   pair (1,4) : d = 0.495 m  →  d/λ = 1.40  →  most ambiguous
#
# Rule: use the SMALLEST unambiguous pair first for sign extraction.
#       For angles in [-60°, +60°] with d=0.165m the wrapped phase is always
#       in [-π/2, +π/2] and its sign directly equals the AoA sign.
# ---------------------------------------------------------------------------

# Primary: pair (2,3) — unambiguous for all collected angles
_SIGN_COLS_PAIR23 = [
    "pair23_tx_corr_phase_mean",     # TX-corrected wrapped phase (unambiguous)
    "pair23_signed_phase_diff_rad",
    "pair23_rx_phase_mean",          # raw wrapped (no TX corr, fallback)
]

# Secondary: pair (1,3) or (2,4) — ambiguous only for |AoA| > ~32°
_SIGN_COLS_PAIR13_24 = [
    "pair13_tx_corr_phase_mean",
    "pair13_signed_phase_diff_rad",
    "pair24_tx_corr_phase_mean",
    "pair24_signed_phase_diff_rad",
]

# Tertiary: pair (1,4) — highly ambiguous, use only as last resort with TX
_SIGN_COLS_PAIR14 = [
    "pair14_tx_corr_phase_unwrap_mean",
    "pair14_tx_corr_phase_mean",
    "pair14_signed_phase_diff_rad",
]

# Raw unwrapped phase (no TX corr) for any pair — least reliable cross-USRP
_SIGN_COLS_RAW_UNWRAP = [
    "pair23_rx_phase_unwrap_mean",
    "pair13_rx_phase_unwrap_mean",
    "pair24_rx_phase_unwrap_mean",
    "pair14_rx_phase_unwrap_mean",
]

# Unprefixed fallback: single-pair scenario (scripts 05-08)
_SIGN_COLS_UNPREFIXED = [
    "tx_corr_phase_mean",
    "signed_phase_diff_rad",
    "rx_phase_mean",
    "tx_corr_phase_unwrap_mean",
    "rx_phase_unwrap_mean",
]


def _get_phase_sign(feature_row):
    """
    Return the consensus sign of the inter-element phase differences.

    Accepts a dict-like object (pandas Series, plain dict, or any mapping).

    Priority order (ranked by phase unambiguity at 850 MHz):
    1. Pair (2,3)  d=0.165m  d/λ=0.47  — ALWAYS unambiguous for [-60°, +60°]
    2. Pair (1,3) or (2,4)  d=0.330m  d/λ=0.93  — ambiguous beyond |AoA|>32°
    3. Pair (1,4) with TX-corrected unwrap  d=0.495m  d/λ=1.40
    4. Raw unwrapped phase (any pair, no TX correction)
    5. Unprefixed single-pair columns

    Returns
    -------
    +1.0  if the consensus phase is positive  → positive AoA
    -1.0  if the consensus phase is negative  → negative AoA
     0.0  if no phase features are found      → no correction applied
    """
    def _collect(col_list):
        vals = []
        for col in col_list:
            if col in feature_row:
                v = feature_row[col]
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(v):
                    vals.append(v)
        return vals

    # Try sources in priority order — stop at the first non-empty group
    for col_list in [
        _SIGN_COLS_PAIR23,
        _SIGN_COLS_PAIR13_24,
        _SIGN_COLS_PAIR14,
        _SIGN_COLS_RAW_UNWRAP,
        _SIGN_COLS_UNPREFIXED,
    ]:
        vals = _collect(col_list)
        if vals:
            # Arithmetic mean (not circular) — these values are all in [-π, +π]
            # for the unambiguous pairs and their sign is directly meaningful
            consensus = float(np.mean(vals))
            return float(np.sign(consensus)) if consensus != 0 else 0.0

    return 0.0  # no information — leave prediction as-is


def apply_sign_correction(pred_angle_deg, feature_row):
    """
    Apply sign correction to a single predicted AoA.

    Parameters
    ----------
    pred_angle_deg : float
        Raw model output (magnitude, typically ≥ 0 because trained on positive
        angles only).
    feature_row : dict-like
        Feature row for the same sample.  Must contain at least one of the
        signed_phase_diff_rad columns.

    Returns
    -------
    float
        Sign-corrected AoA in degrees:
          - positive if phase consensus is positive (AoA ∈ [0°, 60°])
          - negative if phase consensus is negative (AoA ∈ [-60°, 0°])
          - unchanged if no phase features are available
    """
    sign = _get_phase_sign(feature_row)

    if sign > 0:
        return float(abs(pred_angle_deg))
    elif sign < 0:
        return float(-abs(pred_angle_deg))
    else:
        # No phase information — return the model's raw output unchanged
        return float(pred_angle_deg)


def apply_sign_correction_batch(pred_angles_deg, feature_df):
    """
    Apply sign correction to a batch of predicted AoA values.

    Parameters
    ----------
    pred_angles_deg : array-like, shape (N,)
        Raw model outputs.
    feature_df : pandas DataFrame or list of dicts
        Feature rows, one per prediction.

    Returns
    -------
    np.ndarray, shape (N,)
        Sign-corrected AoA values in degrees.
    """
    pred_angles_deg = np.asarray(pred_angles_deg, dtype=float)
    corrected = np.empty_like(pred_angles_deg)

    # Determine iteration method based on input type
    if hasattr(feature_df, "iterrows"):
        rows = [row for _, row in feature_df.iterrows()]
    else:
        rows = feature_df

    for i, row in enumerate(rows):
        corrected[i] = apply_sign_correction(pred_angles_deg[i], row)

    return corrected
