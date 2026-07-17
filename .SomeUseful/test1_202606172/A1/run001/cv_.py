"""Drop-in CV transform/background figures.

Run inside a folder that has ``data.csv`` (columns: scan,time,potential,current;
equilibration rows have blank scan/time). Outputs into the same folder:

  1_current_raw.png        1_current_bkgsub.png
  2_semideriv_raw.png      2_semideriv_bkgsub.png
  3_semiintegral_raw.png   3_semiintegral_bkgsub.png
  overview.png             (all six, 3x2)

Background = per-scan R-CPE fit in the semiderivative domain (current scaled by
1e6 first so lmfit actually iterates at pA-nA scale, then scaled back). Each
domain plots the same physical background-subtracted current, transformed.

Usage:  python cv_figures.py            # uses ./data.csv
        python cv_figures.py <folder>   # uses <folder>/data.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import lmfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

SC = 1e6          # lift pA-nA into ~unity so lmfit doesn't silently converge
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2"]


# ── fractional calculus (Oldham-Spanier RL differintegral) ──────────
def differint(t: np.ndarray, y: np.ndarray, q: float) -> np.ndarray:
    """q-th RL differintegral on an equidistant grid starting at 0.
    q=+0.5 semiderivative, q=-0.5 semi-integral."""
    n = len(t)
    j = np.arange(n)
    quotient = np.zeros(n)
    quotient[1:] = (j[1:] / t[1:]) ** q
    w = np.zeros(n)
    w[0] = 1.0 / gamma(2.0 - q)
    w[1:] = ((j[1:] + 1.0) ** (1.0 - q) - 2.0 * j[1:] ** (1.0 - q)
             + (j[1:] - 1.0) ** (1.0 - q)) / gamma(2.0 - q)
    y_rev = y[::-1]
    s = np.zeros(n)
    for k in range(1, n):
        s[k] = np.dot(w[: k + 1], y_rev[n - 1 - k:])
    return quotient * s


def _vertices(E: np.ndarray) -> np.ndarray:
    """Sweep turning points: where sign(dE) flips."""
    dE = np.diff(E)
    sign = np.sign(dE)
    last = 0.0
    for k in range(len(sign)):
        if sign[k] == 0.0:
            sign[k] = last
        else:
            last = sign[k]
    return np.where((sign[:-1] != sign[1:]) & (sign[1:] != 0.0))[0] + 1


def _piecewise(t, t_discs, values):
    idx = np.clip(np.digitize(t, t_discs[1:-1]), 0, len(values) - 1)
    return np.asarray(values)[idx]


def _make_bg(t, rate, t_discs):
    """Semiderivative of a piecewise R-CPE charging background (closure)."""
    n_seg = len(t_discs) - 1

    def bg(**p):
        Q = _piecewise(t, t_discs, [p[f"Q{k+1}"] for k in range(n_seg)])
        a = _piecewise(t, t_discs, [p[f"a{k+1}"] for k in range(n_seg)])
        cpe = np.zeros_like(t)
        for k in range(1, n_seg + 1):
            with np.errstate(invalid="ignore"):
                t1 = np.where(t >= t_discs[k-1], (t - t_discs[k-1]) ** (1 - a), 0.0)
                t2 = np.where(t >= t_discs[k], (t - t_discs[k]) ** (1 - a), 0.0)
            cpe += (-1) ** (k + 1) * (t1 - t2)
        cpe *= Q * rate / gamma(2 - a)
        return differint(t, cpe, 0.5)

    return bg, n_seg


def fit_background_sd(t, E, i_scaled, rate):
    """Fit R-CPE background in the semiderivative domain (current pre-scaled)."""
    n = len(E)
    sd = differint(t, i_scaled, 0.5)
    v_per_index = float(np.median(np.abs(np.diff(E)))) or 1e-3
    vtx = _vertices(E)
    t_discs = np.concatenate(([0.0], t[vtx], [t[-1]]))
    bg, n_seg = _make_bg(t, rate, t_discs)

    r1 = max(3, int(0.10 / v_per_index))
    r2 = max(2, int(0.05 / v_per_index))
    subset = set(range(0, r1)) | set(range(n - r2, n))
    for v in vtx:
        subset |= set(range(max(0, v - r2), min(n, v + r2)))
    subset = np.array(sorted(subset))

    params = lmfit.Parameters()
    q_guess = abs(i_scaled[min(r1, n - 1)] / rate) + 1e-12
    for k in range(n_seg):
        params.add(f"Q{k+1}", value=q_guess, min=0)
        params.add(f"a{k+1}", value=0.9, min=0.6, max=1.0)

    def resid(p):
        vals = {k: par.value for k, par in p.items()}
        return bg(**vals)[subset] - sd[subset]

    try:
        out = lmfit.minimize(resid, params, method="least_squares", max_nfev=4000)
        fit = {k: float(par.value) for k, par in out.params.items()}
        return bg(**fit)            # background in sd domain (scaled units)
    except Exception:
        return np.zeros_like(sd)


# ── data ────────────────────────────────────────────────────────────
def read_scans(csv_path: Path):
    """data.csv -> {scan: (E, i, t)}; equilibration rows (blank scan) skipped."""
    scans: dict = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not row["scan"]:
                continue
            E, i, t = scans.setdefault(int(row["scan"]), ([], [], []))
            E.append(float(row["potential"]))
            i.append(float(row["current"]))
            t.append(float(row["time"]))
    return {s: tuple(np.asarray(a, float) for a in v) for s, v in scans.items()}


def curves_for_scan(E, i, t):
    """Return the six curves (scaled units; current-domain = uA since SC=1e6)."""
    if t is None or len(t) < 3:
        raise ValueError("scan too short")
    rate = float(np.median(np.abs(np.diff(E))) / np.median(np.diff(t)))  # V/s
    i_s = i * SC
    bkg_sd = fit_background_sd(t, E, i_s, rate)        # background in sd domain
    bkg_cur = differint(t, bkg_sd, -0.5)               # background current
    i_far = i_s - bkg_cur                              # faradaic current
    return {
        "current": (i_s, i_far),
        "semideriv": (differint(t, i_s, 0.5), differint(t, i_far, 0.5)),
        "semiint": (differint(t, i_s, -0.5), differint(t, i_far, -0.5)),
    }


# ── plotting ────────────────────────────────────────────────────────
PANELS = [
    ("current",   0, "1_current_raw",        "Current CV (raw)",                 "i (uA)"),
    ("current",   1, "1_current_bkgsub",     "Current CV (background subtracted)", "i_faradaic (uA)"),
    ("semideriv", 0, "2_semideriv_raw",      "Semiderivative (raw)",             "d^1/2 i / dt^1/2 (a.u.)"),
    ("semideriv", 1, "2_semideriv_bkgsub",   "Semiderivative (bkg subtracted = residual)", "d^1/2 i / dt^1/2 (a.u.)"),
    ("semiint",   0, "3_semiintegral_raw",   "Semi-integral (raw)",              "I_-1/2 (a.u.)"),
    ("semiint",   1, "3_semiintegral_bkgsub","Semi-integral (background subtracted)", "I_-1/2 (a.u.)"),
]


def main(folder="."):
    folder = Path(folder)
    csv_path = folder / "data.csv"
    if not csv_path.exists():
        sys.exit(f"no data.csv in {folder.resolve()}")
    scans = read_scans(csv_path)
    if not scans:
        sys.exit("data.csv has no CV scans (only equilibration?)")

    per_scan = {}
    for s in sorted(scans):
        try:
            per_scan[s] = (scans[s][0], curves_for_scan(*scans[s]))
        except Exception as exc:
            print(f"scan {s} skipped: {exc}")

    def draw(ax, key, col, ylabel, title):
        for n, s in enumerate(sorted(per_scan)):
            E, cur = per_scan[s]
            ax.plot(E, cur[key][col], lw=1, color=COLORS[n % len(COLORS)],
                    label=f"scan {s}")
        ax.axhline(0, color="#bbb", lw=0.6)
        ax.set_xlabel("E (V)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        if len(per_scan) > 1:
            ax.legend(fontsize=8, frameon=False)

    # individual figures
    for key, col, fname, title, ylabel in PANELS:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        draw(ax, key, col, ylabel, title)
        fig.tight_layout()
        fig.savefig(folder / f"{fname}.png", dpi=130)
        plt.close(fig)

    # combined overview
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for ax, (key, col, _f, title, ylabel) in zip(axes.flat, PANELS):
        draw(ax, key, col, ylabel, title)
    fig.tight_layout()
    fig.savefig(folder / "overview.png", dpi=130)
    plt.close(fig)

    print(f"done -> {folder.resolve()} (7 png: 6 individual + overview)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")