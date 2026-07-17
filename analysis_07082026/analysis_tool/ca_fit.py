"""CA fitting layer: multi-start power-law+drift fit, adaptive tail cut, gates, failure classing.
Reads result/manifest.csv (from build_manifest) to pair each CA with its background and own CV."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

BETA_BOUNDS = (0.01, 1.5)
BETA_STARTS = (0.3, 0.5, 0.8)
WIN = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
REF_T_MIN = 0.3
K_CUT = 3.0
BIN_SECONDS = 1.0
SNR_MIN = 20.0
GATE_SPREAD = 0.02
GATE_PLATEAU = 0.02
EXCESS_Z_FLAG = 4.0
GATE_A_DEV = 0.05
BLANK_K = 3.0


def load_ca(path):
    """Read a CA data.csv into (potential_v, t, i)."""
    with open(path, encoding="utf-8") as f:
        head = f.readline()
        next(f)
        rows = [line.rstrip("\n").split(",") for line in f]
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", head)
    potential = float(m.group(0)) if m else np.nan
    t = np.array([float(r[0]) for r in rows])
    i = np.array([float(r[1]) for r in rows])
    return potential, t, i


def robust_mad(x):
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def succ_sigma(x):
    d = np.diff(x)
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2))


def model(t, a, beta, c, m):
    """Power-law decay with offset and linear drift."""
    return a * t ** (-beta) + c + m * t


def fit_window(t, i):
    """Multi-start fit; returns params dict with beta_spread, or None."""
    a0 = i[0] * t[0] ** 0.5
    c0 = float(np.mean(i[-max(5, len(i) // 10):]))
    fits = []
    for b0 in BETA_STARTS:
        try:
            popt, _ = curve_fit(model, t, i, p0=[a0, b0, c0, 0.0],
                                bounds=([-np.inf, BETA_BOUNDS[0], -np.inf, -np.inf],
                                        [np.inf, BETA_BOUNDS[1], np.inf, np.inf]),
                                maxfev=20000)
            yfit = model(t, *popt)
            fits.append((float(np.sum((i - yfit) ** 2)), popt, yfit))
        except (RuntimeError, ValueError):
            continue
    if not fits:
        return None
    ssr, popt, yfit = min(fits, key=lambda f: f[0])
    betas = [f[1][1] for f in fits]
    return {"A": popt[0], "beta": popt[1], "C": popt[2], "M": popt[3],
            "beta_spread": max(betas) - min(betas),
            "r2": 1 - ssr / float(np.sum((i - np.mean(i)) ** 2)),
            "resid": i - yfit}


def adaptive_tmax(t, i, sigma, k=K_CUT, max_iter=5):
    """Tail cut where the power-law signal drowns in drift or noise."""
    t_max = t[-1]
    for _ in range(max_iter):
        m = (t >= REF_T_MIN) & (t <= t_max)
        f = fit_window(t[m], i[m])
        if f is None or abs(f["A"]) <= 0:
            return t[-1]
        cut_d = ((abs(f["A"]) / (k * abs(f["M"]))) ** (1.0 / (1.0 + f["beta"]))
                 if abs(f["M"]) > 0 else np.inf)
        log_n = np.log(abs(f["A"]) / (k * sigma)) / f["beta"] if sigma > 0 else 20
        cut_n = np.exp(log_n) if log_n < 20 else np.inf
        t_cut = min(max(min(cut_d, cut_n), 10 * REF_T_MIN), t[-1])
        if abs(t_cut - t_max) / t_max < 0.05:
            return t_cut
        t_max = t_cut
    return t_max


def bin_excess(t, r):
    """Structure amplitude of 1s-binned residual means vs white expectation."""
    idx = np.floor((t - t[0]) / BIN_SECONDS).astype(int)
    counts = np.bincount(idx)
    keep = counts > 0
    rb = np.bincount(idx, weights=r)[keep] / counts[keep]
    n_bin = max(1, int(round(len(r) / max(1, len(rb)))))
    white = succ_sigma(r) / np.sqrt(n_bin)
    return robust_mad(rb) / white if white > 0 else np.nan


def fit_ca(t, i):
    """Full CA fit: sigma, tail cut, plateau over the t_min window, diagnostics."""
    sigma = succ_sigma(i[len(i) // 2:])
    t_max = adaptive_tmax(t, i, sigma)
    fits = {}
    for t_min in WIN:
        m = (t >= t_min) & (t <= t_max)
        if m.sum() < 8:
            continue
        f = fit_window(t[m], i[m])
        if f is not None:
            fits[t_min] = f
    if REF_T_MIN not in fits:
        return None
    ref = fits[REF_T_MIN]
    betas = [f["beta"] for f in fits.values()]
    mref = (t >= REF_T_MIN) & (t <= t_max)
    return {"beta": float(np.median(betas)),
            "beta_rng": max(betas) - min(betas),
            "beta_spread_max": max(f["beta_spread"] for f in fits.values()),
            "A": ref["A"], "C": ref["C"], "M": ref["M"], "r2": ref["r2"],
            "sigma_noise": sigma, "t_max_used": t_max,
            "snr": abs(ref["A"]) * REF_T_MIN ** (-ref["beta"]) / sigma if sigma > 0 else np.nan,
            "bin_excess": bin_excess(t[mref], ref["resid"]),
            "mad_ratio": robust_mad(ref["resid"]) / sigma if sigma > 0 else np.nan}


def classify(row):
    """Failure cause from the gate pattern and the M-sign fingerprint."""
    if not row["gate_ident"]:
        return "beta_unidentifiable"
    if not row["gate_signal"]:
        return "no_signal_or_depleted"
    if not row["gate_plateau"]:
        return "unsettled_fresh" if row["M"] < 0 else "post_cv_unrelaxed"
    if not row["gate_A"]:
        return "electrode_or_depletion"
    return "ok"


def gate_rows(rows):
    """Gates (ident, plateau, signal, A-consistency), analyte call, verdict; bin_excess diagnostic only."""
    for sol in {r["solution"] for r in rows}:
        grp = [r for r in rows if r["solution"] == sol and np.isfinite(r["A"])]
        med = np.median([r["A"] for r in grp]) if len(grp) >= 2 else np.nan
        for r in grp:
            r["A_dev"] = abs(r["A"] - med) / abs(med) if np.isfinite(med) and med != 0 else np.nan
    blanks = [r["A"] for r in rows if r["solution"].startswith("blank") and np.isfinite(r["A"])]
    a_thr = (np.median(blanks) + BLANK_K * robust_mad(np.array(blanks))) if len(blanks) >= 2 else np.nan
    blank_exc = [r["bin_excess"] for r in rows if r["solution"].startswith("blank")
                 and np.isfinite(r["bin_excess"])]
    exc_floor = float(np.median(blank_exc)) if blank_exc else np.nan
    exc_scale = float(robust_mad(np.array(blank_exc))) if len(blank_exc) >= 2 else np.nan
    for r in rows:
        r.setdefault("A_dev", np.nan)
        r["gate_signal"] = bool(r["snr"] >= SNR_MIN) if np.isfinite(r["snr"]) else False
        r["gate_ident"] = bool(r["beta_spread_max"] <= GATE_SPREAD)
        r["gate_plateau"] = bool(r["beta_rng"] <= GATE_PLATEAU)
        r["gate_A"] = bool(r["A_dev"] <= GATE_A_DEV) if np.isfinite(r["A_dev"]) else True
        r["bin_excess_z"] = ((r["bin_excess"] - exc_floor) / max(exc_scale, 0.3)
                             if np.isfinite(r["bin_excess"]) and np.isfinite(exc_floor) else np.nan)
        r["bg_flag"] = bool(r["bin_excess_z"] > EXCESS_Z_FLAG) if np.isfinite(r["bin_excess_z"]) else False
        quality = r["gate_ident"] and r["gate_plateau"] and r["gate_A"]
        r["pass_all"] = quality and r["gate_signal"]
        r["failure_class"] = classify(r)
        r["analyte_present"] = bool(abs(r["A"]) > a_thr) if np.isfinite(a_thr) else None
        if r["solution"].startswith("blank"):
            r["verdict"] = "contaminated_blank" if r["analyte_present"] else "clean_blank"
        else:
            r["verdict"] = "good" if r["pass_all"] else r["failure_class"]
    return rows


def _read_manifest(iso_dir):
    """Load result/manifest.csv rows."""
    mpath = Path(iso_dir) / "result" / "manifest.csv"
    if not mpath.is_file():
        raise FileNotFoundError(f"{mpath} not found; run build_manifest first")
    with open(mpath, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run(iso_dir):
    """Fit every CA in the manifest, gate, classify; write result/ca_gates.csv. Returns rows."""
    iso = Path(iso_dir)
    man = _read_manifest(iso)
    rows = []
    for m in man:
        if not m["ca"]:
            continue
        sol = "blank" if m["role"] in ("baseline", "blank") else "sample"
        potential, t, i = load_ca(iso / m["ca"] / "data.csv")
        f = fit_ca(t, i)
        row = {"file": m["ca"], "solution": sol, "role": m["role"],
               "bg_ca": m["bg_ca"], "own_cv": m["cv"], "potential": potential}
        if f is None:
            row.update({k: np.nan for k in ("beta", "beta_rng", "beta_spread_max", "A", "C", "M",
                                            "r2", "sigma_noise", "t_max_used", "snr",
                                            "bin_excess", "mad_ratio")})
        else:
            f.pop("resid", None)
            row.update(f)
        rows.append(row)
    rows = gate_rows(rows)
    out = iso / "result" / "ca_gates.csv"
    fields = sorted({k for r in rows for k in r})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} CAs -> {out}")
    return rows


def session_betas(iso_dir):
    """{own_cv_dir: certified beta} for feeding CV order; pass_all sample CAs only."""
    rows = run(iso_dir)
    return {r["own_cv"]: r["beta"] for r in rows
            if r.get("pass_all") and r["solution"] == "sample" and r["own_cv"]}