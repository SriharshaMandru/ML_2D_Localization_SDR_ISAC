import numpy as np

C = 3e8

# ---------------------------------------------------------------------------
# Antenna layout (physical left → right):
#   Ant1 (USRP1-RX2) | Ant2 (USRP1-TxRx) | Ant3 (USRP2-RX2) | Ant4 (USRP2-TxRx)
#
# Sign convention (ULA far-field model):
#   phase_diff = angle(x_right * conj(x_left)) = (2π·d/λ)·sin(θ)
#   → positive phase_diff  ↔  positive AoA (signal from the right)
#   → negative phase_diff  ↔  negative AoA (signal from the left)
#
# Cross-USRP pairs (1,3) (1,4) (2,3) (2,4) have a random inter-LO phase
# offset. The TX reference is used to remove this hardware bias so that the
# sign of signed_phase_diff_rad purely reflects the wavefront direction.
# ---------------------------------------------------------------------------


def wavelength(freq_mhz):
    return C / (float(freq_mhz) * 1e6)


def power_db(x):
    return 10 * np.log10(np.mean(np.abs(x) ** 2) + 1e-12)


def wrap_angle(a):
    """Wrap angle in radians to [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def tx_phase_offset(tx, rx):
    """
    Return the mean LO/cable phase offset of a single RX branch relative to TX.

    Uses circular mean so it is robust to phase wrapping.
    Preserves sign — does NOT take abs.
    """
    n = min(len(tx), len(rx))
    cross = rx[:n] * np.conj(tx[:n])
    return float(np.angle(np.mean(np.exp(1j * np.angle(cross)))))


def signed_phase_to_aoa_deg(phase_rad, freq_mhz, baseline_m):
    """
    Convert a **signed** inter-element phase difference (radians) to AoA (degrees).

    Sign contract:
        phase_rad > 0  →  AoA > 0  (signal from positive/right side)
        phase_rad < 0  →  AoA < 0  (signal from negative/left side)
        phase_rad = 0  →  AoA = 0  (broadside)

    Uses arcsin which natively preserves sign.
    """
    lam = wavelength(freq_mhz)
    value = (phase_rad * lam) / (2 * np.pi * baseline_m)
    value = np.clip(value, -1.0, 1.0)
    return np.degrees(np.arcsin(value))


# Keep the old name as an alias so existing callers are not broken.
phase_to_aoa_deg = signed_phase_to_aoa_deg


def estimate_corr_lag(x, y, max_lag=128):
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    best_lag = 0
    best_corr = -1.0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = x[-lag:]
            b = y[: len(a)]
        elif lag > 0:
            a = x[:-lag]
            b = y[lag:]
        else:
            a = x
            b = y

        if len(a) < 32:
            continue

        corr = np.abs(np.vdot(a, b))
        corr = corr / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

        if corr > best_corr:
            best_corr = float(corr)
            best_lag = int(lag)

    return best_lag, best_corr


def tx_rx_features(tx, rx, prefix):
    n = min(len(tx), len(rx))
    tx = tx[:n]
    rx = rx[:n]

    lag, corr_peak = estimate_corr_lag(tx, rx)

    cross = rx * np.conj(tx)
    phase = np.angle(cross)

    return {
        f"{prefix}_txrx_corr_lag": lag,
        f"{prefix}_txrx_corr_peak": corr_peak,
        f"{prefix}_txrx_phase_mean": float(np.angle(np.mean(np.exp(1j * phase)))),
        f"{prefix}_txrx_phase_std": float(np.std(phase)),
        f"{prefix}_txrx_power_ratio": float(
            np.mean(np.abs(rx) ** 2) / (np.mean(np.abs(tx) ** 2) + 1e-12)
        ),
        f"{prefix}_txrx_rssi_diff_db": float(power_db(rx) - power_db(tx)),
    }


def extract_pair_features(x1, x2, tx, freq_mhz, baseline_m):
    """
    Extract features for one antenna pair.

    x1  : complex IQ for the LEFT  antenna of the pair (lower index)
    x2  : complex IQ for the RIGHT antenna of the pair (higher index)
    tx  : TX reference IQ (may be None)

    The signed inter-element phase difference is computed as:
        phase_diff = angle(x2 * conj(x1))
    which follows the ULA model  phase_diff = (2π·d/λ)·sin(θ).

    When TX is available the per-branch hardware offsets are subtracted:
        offset_k = angle( rx_k * conj(tx) )  [LO / cable bias of branch k]
        hw_correction = offset2 - offset1
        signed_phase_diff = wrap( raw_mean_phase - hw_correction )

    This ensures that the sign of signed_phase_diff_rad purely reflects the
    direction of arrival, not SDR/cable artefacts.
    """
    amp1 = np.abs(x1)
    amp2 = np.abs(x2)

    rssi1 = power_db(x1)
    rssi2 = power_db(x2)

    # --- Raw inter-element cross product (x2 leads for positive AoA) --------
    # The raw cross product at each sample:
    #   cross[n] = x2[n] * conj(x1[n])
    # Its angle is: phase[n] = phi2[n] - phi1[n]
    #             = (carrier + hw_offset2 + aoa_delay) - (carrier + hw_offset1)
    #             = (hw_offset2 - hw_offset1) + aoa_delay
    # So the hardware offsets appear as a constant additive bias on every sample.
    cross = x2 * np.conj(x1)
    phase = np.angle(cross)                   # wrapped, [-π, π] per sample
    phase_unwrapped = np.unwrap(phase)        # raw unwrapped (no TX correction)
    
    # --- TX-based hardware phase-offset correction — two-step -----------------
    # Step 1: TX-normalise both branches to remove the shared carrier phase.
    #   x_k_norm[n] = rx_k[n] / tx[n]
    #   Phase of x_k_norm[n] ≈ hw_offset_k  (constant; no more carrier)
    #
    # Step 2: Subtract the differential hardware offset from the inter-element
    #   phase to obtain the pure spatial (AoA) phase difference:
    #   phase_spatial = phase(x2_norm * conj(x1_norm)) - 0   [= aoa_delay only]
    #
    # After Step 1 the cross product is:
    #   cross_norm[n] = x2_norm[n] * conj(x1_norm[n])
    #   angle ≈ (hw_offset2 - hw_offset1) + aoa_delay
    # So the mean angle is: mean_phase_norm ≈ (hw2 - hw1) + aoa_delay
    # And the per-branch TX offsets (measured from the normalised signals) are:
    #   mean_angle(x1_norm) ≈ hw_offset1
    #   mean_angle(x2_norm) ≈ hw_offset2
    # Therefore:
    #   aoa_phase = mean_phase_norm - (hw_offset2 - hw_offset1)
    if tx is not None:
        n_common = min(len(tx), len(x1), len(x2))
        tx_c = tx[:n_common]

        # Step 1: TX normalisation — removes common carrier
        x1_norm = x1[:n_common] / (tx_c + 1e-30)
        x2_norm = x2[:n_common] / (tx_c + 1e-30)

        # Step 2a: inter-element cross on the TX-normalised signals
        cross_corr = x2_norm * np.conj(x1_norm)
        phase_corr = np.angle(cross_corr)
        mean_phase_norm = float(np.angle(np.mean(np.exp(1j * phase_corr))))

        # Step 2b: residual per-branch hardware offsets (= mean angle of each norm signal)
        hw1_est = float(np.angle(np.mean(np.exp(1j * np.angle(x1_norm)))))
        hw2_est = float(np.angle(np.mean(np.exp(1j * np.angle(x2_norm)))))

        # Step 2c: subtract differential hardware offset → pure spatial phase
        hw_diff = hw2_est - hw1_est
        mean_phase_corr = float(wrap_angle(mean_phase_norm - hw_diff))
    else:
        # No TX — use raw cross product; hardware offsets remain as bias
        cross_corr = cross
        phase_corr = phase
        mean_phase_corr = float(np.angle(np.mean(np.exp(1j * phase_corr))))

    phase_corr_unwrap = np.unwrap(phase_corr)    # continuous
    mean_phase_corr_unwrap = float(np.mean(phase_corr_unwrap))

    # Primary signed phase diff:
    #   - magnitude from the wrapped circular mean (what arcsin expects, in [-π,π])
    #   - sign from the unwrapped mean (reliable when d/λ < ~0.5 and N is large)
    # For the unambiguous pair (2,3) at 850 MHz (d/λ=0.47), the wrapped mean
    # itself always has the correct sign.  For large baselines the sign from
    # unwrap may still be wrong; sign_correct_aoa.py always prefers pair (2,3).
    sign_from_unwrap = np.sign(mean_phase_corr_unwrap) if mean_phase_corr_unwrap != 0 else 1.0
    signed_phase_diff = float(abs(mean_phase_corr) * sign_from_unwrap)

    # AoA derived from the corrected, signed phase
    signed_aoa_deg = float(signed_phase_to_aoa_deg(signed_phase_diff, freq_mhz, baseline_m))

    # --- Spatial covariance matrix entries ---------------------------------
    r11 = np.mean(x1 * np.conj(x1))
    r22 = np.mean(x2 * np.conj(x2))
    r12 = np.mean(x1 * np.conj(x2))

    rx_lag, rx_corr_peak = estimate_corr_lag(x1, x2)

    features = {
        "rssi1_db": float(rssi1),
        "rssi2_db": float(rssi2),
        "rssi_avg_db": float((rssi1 + rssi2) / 2.0),
        "rssi_diff_db": float(rssi1 - rssi2),
        "power_ratio_rx1_rx2": float(
            np.mean(amp1 ** 2) / (np.mean(amp2 ** 2) + 1e-12)
        ),
        "amp1_mean": float(np.mean(amp1)),
        "amp1_std": float(np.std(amp1)),
        "amp1_max": float(np.max(amp1)),
        "amp2_mean": float(np.mean(amp2)),
        "amp2_std": float(np.std(amp2)),
        "amp2_max": float(np.max(amp2)),
        # ── Raw (uncorrected) phase statistics — kept for backward compat ──
        "rx_phase_mean": float(np.angle(np.mean(np.exp(1j * phase)))),
        "rx_phase_median": float(np.median(phase)),
        "rx_phase_std": float(np.std(phase)),
        "rx_phase_unwrap_mean": float(np.mean(phase_unwrapped)),
        "rx_phase_unwrap_std": float(np.std(phase_unwrapped)),
        # ── TX-corrected phase statistics ─────────────────────────────────
        # signed_phase_diff_rad  : sign-correct inter-element phase
        #                          (abs from wrapped circ-mean, sign from unwrap)
        # signed_phase_diff_aoa_deg : corresponding AoA via arcsin
        # tx_corr_phase_mean     : TX-corrected circular mean (wrapped) — for ML
        # tx_corr_phase_unwrap_mean : TX-corrected unwrapped mean — for sign detection
        "signed_phase_diff_rad": signed_phase_diff,
        "signed_phase_diff_aoa_deg": signed_aoa_deg,
        "tx_corr_phase_mean": mean_phase_corr,
        "tx_corr_phase_unwrap_mean": mean_phase_corr_unwrap,
        # Legacy name — kept so downstream code reading phase_aoa_deg still works
        "phase_aoa_deg": signed_aoa_deg,
        "rx_corr_lag": rx_lag,
        "rx_corr_peak": rx_corr_peak,
        "scm_R11_real": float(np.real(r11)),
        "scm_R22_real": float(np.real(r22)),
        "scm_R12_real": float(np.real(r12)),
        "scm_R12_imag": float(np.imag(r12)),
        "scm_R12_abs": float(np.abs(r12)),
        "scm_R12_phase": float(np.angle(r12)),
    }

    if tx is not None:
        features.update(tx_rx_features(tx, x1, "rx1"))
        features.update(tx_rx_features(tx, x2, "rx2"))

    return features
