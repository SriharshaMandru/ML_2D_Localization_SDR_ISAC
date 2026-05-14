import numpy as np


def rssi_to_distance(rssi_db, p0_db=-40.0, n=2.0, d0=1.0):
    return d0 * 10 ** ((p0_db - rssi_db) / (10 * n))


def distance_to_rssi(distance_m, p0_db=-40.0, n=2.0, d0=1.0):
    distance_m = np.maximum(distance_m, 1e-9)
    return p0_db - 10 * n * np.log10(distance_m / d0)
