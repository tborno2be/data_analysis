"""Fit CA decay curves with power-law models over a grid of start times."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

T_MINS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5]
BETA_BOUNDS = (0.01, 1.5)


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


def fit_one(t, i, with_drift):
    """Fit one model to (t, i); return dict of parameters and r2, NaN on failure."""
    a0 = i[0] * t[0] ** 0.5
    c0 = float(np.mean(i[-max(5, len(i) // 10):]))
    out = {"A": np.nan, "beta": np.nan, "C": np.nan, "M": np.nan, "r2": np.nan}
    try:
        if with_drift:
            p0 = [a0, 0.5, c0, 0.0]
            lo = [-np.inf, BETA_BOUNDS[0], -np.inf, -np.inf]
            hi = [np.inf, BETA_BOUNDS[1], np.inf, np.inf]
            popt, _ = curve_fit(model_linear, t, i, p0=p0, bounds=(lo, hi), maxfev=20000)
            out.update(A=popt[0], beta=popt[1], C=popt[2], M=popt[3],
                       r2=r_squared(i, model_linear(t, *popt)))
        else:
            p0 = [a0, 0.5, c0]
            lo = [-np.inf, BETA_BOUNDS[0], -np.inf]
            hi = [np.inf, BETA_BOUNDS[1], np.inf]
            popt, _ = curve_fit(model_simple, t, i, p0=p0, bounds=(lo, hi), maxfev=20000)
            out.update(A=popt[0], beta=popt[1], C=popt[2],
                       r2=r_squared(i, model_simple(t, *popt)))
    except (RuntimeError, ValueError):
        pass
    return out


def main(folder, out_csv="ca_fit_grid.csv"):
    """Run the full fitting grid and write a long-format summary CSV."""
    records = []
    for path in find_ca_files(folder):
        potential, t, i = load_ca(path)
        for t_min in T_MINS:
            mask = t >= t_min
            ts, cs = t[mask], i[mask]
            for model, drift in (("simple", False), ("linear", True)):
                if len(ts) < (5 if drift else 4):
                    res = {"A": np.nan, "beta": np.nan, "C": np.nan,
                           "M": np.nan, "r2": np.nan}
                else:
                    res = fit_one(ts, cs, drift)
                records.append({"file": path.parent.name, "potential": potential,
                                "t_min": t_min, "model": model,
                                **res, "n_points": len(ts)})
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"{len(df)} rows -> {out_csv}")
    return df


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")