import os
import numpy as np
import matplotlib.pyplot as plt


def save_true_pred_plot(y_true, y_pred, out_path, xlabel, ylabel, title):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.scatter(y_true, y_pred, s=8)

    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))

    plt.plot([lo, hi], [lo, hi])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_cdf(errors, out_path, title="Localization Error CDF"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    e = np.sort(errors)
    y = np.arange(1, len(e) + 1) / len(e)

    plt.figure()
    plt.plot(e, y)
    plt.xlabel("Error")
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
