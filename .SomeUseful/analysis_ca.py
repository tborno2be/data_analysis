"""Fit CA decay with adaptive tail cut and residual-structure diagnostics."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T_MINS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5]
BETA_BOUNDS = (0.01, 1.5)
K_CUT = 3.0
REF_T_MIN = 0.3
BIN_SECONDS = 1.0


def natural_key(p):
    """Sort key extracting the trailing number of a Ca directory name."""
    m = re.search(r"(\d+)$", p.name)
    return int(m.group(1)) if m else 0


def find_ca_files(folder):
    """Return data.csv paths inside Ca* directories, naturally sorted."""
    dirs = [p for p in Path(folder).iterdir()
            if p.is_dir() and p.name.lower().startswith("ca")]
    return [d / "data.csv" for d in sorted(dirs, key=natural_key)
            if (d / "data.csv").is_file()]


def parse_potential(line):
    """Extract a float potential from the first line, else return raw text."""
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
    return float(m.group(0)) if m else line.strip()


def load_ca(path):
    """Return (potential, t, i) from a CA file with potential/header/data rows."""
    with open(path) as f:
        lines = f.read().splitlines()
    potential = parse_potential(lines[0])
    sep = "," if "," in lines[2] else None
    rows = []
    for ln in lines[2:]:
        parts = ln.split(sep) if sep else ln.split()
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    arr = np.asarray(rows)
    return potential, arr[:, 0], arr[:, 1]


def model_simple(t, a, beta, c):
    """Power-law decay with offset."""
    return a * t ** (-beta) + c


def model_linear(t, a, beta, c, m):
    """Power-law decay with offset and linear drift."""
    return a * t ** (-beta) + c + m * t


def r_squared(y, y_fit):
    """Coefficient of determination."""
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def robust_mad(x):
    """Scaled median absolute deviation."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def succ_sigma(x):
    """Noise sigma from successive differences via MAD."""
    d = np.diff(x)
    return robust_mad(d) / np.sqrt(2)


def runs_test(r):
    """Wald-Wolfowitz runs test on residual signs; returns (z, p)."""
    s = np.sign(r)
    s = s[s != 0]
    n1, n2 = np.sum(s > 0), np.sum(s < 0)
    n = n1 + n2
    if n1 == 0 or n2 == 0 or n < 10:
        return np.nan, np.nan
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n ** 2 * (n - 1))
    z = (runs - mu) / np.sqrt(var)
    return z, 2 * norm.sf(abs(z))


def lag1(r):
    """Lag-1 autocorrelation."""
    r = r - np.mean(r)
    denom = np.sum(r * r)
    return float(np.sum(r[:-1] * r[1:]) / denom) if denom > 0 else np.nan


def bin_means(t, r, width):
    """Bin residuals by time width; returns (t_bin, r_bin)."""
    idx = np.floor((t - t[0]) / width).astype(int)
    counts = np.bincount(idx)
    keep = counts > 0
    tb = np.bincount(idx, weights=t)[keep] / counts[keep]
    rb = np.bincount(idx, weights=r)[keep] / counts[keep]
    return tb, rb


def fit_one(t, i, with_drift):
    """Fit one model; returns (params dict, fitted curve or None)."""
    a0 = i[0] * t[0] ** 0.5
    c0 = float(np.mean(i[-max(5, len(i) // 10):]))
    out = {"A": np.nan, "beta": np.nan, "C": np.nan, "M": np.nan, "r2": np.nan}
    try:
        if with_drift:
            p0 = [a0, 0.5, c0, 0.0]
            lo = [-np.inf, BETA_BOUNDS[0], -np.inf, -np.inf]
            hi = [np.inf, BETA_BOUNDS[1], np.inf, np.inf]
            popt, _ = curve_fit(model_linear, t, i, p0=p0, bounds=(lo, hi), maxfev=20000)
            yfit = model_linear(t, *popt)
            out.update(A=popt[0], beta=popt[1], C=popt[2], M=popt[3],
                       r2=r_squared(i, yfit))
        else:
            p0 = [a0, 0.5, c0]
            lo = [-np.inf, BETA_BOUNDS[0], -np.inf]
            hi = [np.inf, BETA_BOUNDS[1], np.inf]
            popt, _ = curve_fit(model_simple, t, i, p0=p0, bounds=(lo, hi), maxfev=20000)
            yfit = model_simple(t, *popt)
            out.update(A=popt[0], beta=popt[1], C=popt[2],
                       r2=r_squared(i, yfit))
        return out, yfit
    except (RuntimeError, ValueError):
        return out, None


def adaptive_tmax(t, i, t_min=REF_T_MIN, k=K_CUT, max_iter=5):
    """Iterate t_max to where the power-law signal falls below k*noise."""
    sigma = succ_sigma(i[len(i)//2:])
    t_max = t[-1]
    for _ in range(max_iter):
        m = (t >= t_min) & (t <= t_max)
        res, _ = fit_one(t[m], i[m], True)
        if np.isnan(res["beta"]) or abs(res["A"]) <= 0:
            return t[-1], sigma
        t_cut_drift = (abs(res["A"]) / (k * abs(res["M"]))) ** (1.0 / (1.0 + res["beta"])) \
            if abs(res["M"]) > 0 else np.inf
        log_cut = np.log(abs(res["A"]) / (k * sigma)) / res["beta"]
        t_cut_noise = np.exp(log_cut) if log_cut < 20 else np.inf
        t_cut = min(t_cut_drift, t_cut_noise)
        t_cut = min(max(t_cut, 10 * t_min), t[-1])
        if abs(t_cut - t_max) / t_max < 0.05:
            return t_cut, sigma
        t_max = t_cut
    return t_max, sigma


def resid_diag(t, r, sigma):
    """Residual structure metrics on raw and 1s-binned residuals."""
    
    mad = robust_mad(r)
    _, rb = bin_means(t, r, BIN_SECONDS)
    z, p = runs_test(rb)
    n_bin = max(1, int(round(len(r) / max(1, len(rb)))))
    sigma_eff = succ_sigma(r)
    white_pred = sigma_eff / np.sqrt(n_bin)
    excess = robust_mad(rb) / white_pred if white_pred > 0 else np.nan
    return {"resid_mad": mad,
            "mad_ratio": mad / sigma if sigma > 0 else np.nan,
            "runs_z": z, "runs_p": p, "rho1_bin": lag1(rb), "bin_excess": excess}


def plot_residuals(name, t, r_s, r_l, folder):
    """Save simple-vs-linear residual comparison with binned means."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for ax, r, label in ((axes[0], r_s, "simple"), (axes[1], r_l, "linear")):
        ax.plot(t, r, ".", ms=1, alpha=0.25, color="gray")
        tb, rb = bin_means(t, r, BIN_SECONDS)
        ax.plot(tb, rb, "-", color="crimson", lw=1.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(f"residual ({label}) / A")
    axes[1].set_xlabel("t / s")
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(Path(folder) / f"{name}_resid.png", dpi=150)
    plt.close(fig)


def main(folder, out_csv="ca_fit_grid_v2.csv", plot_dir="resid_plots"):
    """Run the fitting grid with adaptive tail cut and residual diagnostics."""
    Path(plot_dir).mkdir(exist_ok=True)
    records = []
    for path in find_ca_files(folder):
        potential, t, i = load_ca(path)
        t_max, sigma = adaptive_tmax(t, i)
        resid_pair = {}
        for t_min in T_MINS:
            mask = (t >= t_min) & (t <= t_max)
            ts, cs = t[mask], i[mask]
            for model, drift in (("simple", False), ("linear", True)):
                if len(ts) < (5 if drift else 4):
                    res, yfit = ({"A": np.nan, "beta": np.nan, "C": np.nan,
                                  "M": np.nan, "r2": np.nan}, None)
                else:
                    res, yfit = fit_one(ts, cs, drift)
                diag = (resid_diag(ts, cs - yfit, sigma) if yfit is not None
                        else {"resid_mad": np.nan, "mad_ratio": np.nan,
                              "runs_z": np.nan, "runs_p": np.nan,
                              "rho1_bin": np.nan})
                if t_min == REF_T_MIN and yfit is not None:
                    resid_pair[model] = (ts, cs - yfit)
                records.append({"file": path.parent.name, "potential": potential,
                                "t_min": t_min, "model": model, **res,
                                "n_points": len(ts), "t_max_used": t_max,
                                "sigma_noise": sigma, **diag})
        if len(resid_pair) == 2:
            plot_residuals(path.parent.name, resid_pair["simple"][0],
                           resid_pair["simple"][1], resid_pair["linear"][1],
                           plot_dir)
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"{len(df)} rows -> {out_csv}; plots -> {plot_dir}/")
    return df


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")