"""Fit the 115-set and test_iso CA data with identical code, window, and gates."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_ca import (T_MINS, REF_T_MIN, load_ca, fit_one, adaptive_tmax,
                         resid_diag)

XLSX = "Book1.xlsx"
T_MAX_FORCE = 3.0
WIN = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
EMPTY = {"A": np.nan, "beta": np.nan, "C": np.nan, "M": np.nan,
         "r2": np.nan, "beta_spread": np.nan}
EMPTY_DIAG = {"resid_mad": np.nan, "mad_ratio": np.nan, "runs_z": np.nan,
              "runs_p": np.nan, "rho1_bin": np.nan, "bin_excess": np.nan}


def select_115(xlsx):
    """Return metadata for Sample/Blank groups without electrode disconnect."""
    meta = pd.read_excel(xlsx, sheet_name="Sheet1")
    meta = meta[meta["物质"].isin(["Sample", "Blank"])
                & (meta["断电极？"] == "X")]
    return meta.dropna(subset=["组数"]).astype({"组数": int})


def find_testiso_dir(root):
    """Locate the test_iso subfolder regardless of exact spelling."""
    for p in Path(root).iterdir():
        if p.is_dir() and p.name.lower().replace("_", "").startswith("testiso"):
            return p
    return None


def iter_files(root, meta):
    """Yield (dataset, group, path) for both datasets."""
    for n in meta["组数"]:
        yield "ca115", int(n), Path(root) / "CA" / str(n) / "data.csv"
    tdir = find_testiso_dir(root)
    if tdir is not None:
        for n in range(1, 22):
            yield "test_iso", n, tdir / f"Ca{n}" / "data.csv"


def run_file(path):
    """Run the fitting grid on one file; returns list of record dicts or None."""
    if not path.is_file():
        return None
    potential_file, t, i = load_ca(path)
    t_max, sigma = adaptive_tmax(t, i)
    t_max = min(t_max, T_MAX_FORCE)
    records = []
    for t_min in T_MINS:
        mask = (t >= t_min) & (t <= t_max)
        ts, cs = t[mask], i[mask]
        for model, drift in (("simple", False), ("linear", True)):
            if len(ts) < (5 if drift else 4):
                res, yfit = dict(EMPTY), None
            else:
                res, yfit = fit_one(ts, cs, drift)
            diag = (resid_diag(ts, cs - yfit, sigma) if yfit is not None
                    else dict(EMPTY_DIAG))
            records.append({"potential_file": potential_file, "t_min": t_min,
                            "model": model, **res, "n_points": len(ts),
                            "t_max_used": t_max, "sigma_noise": sigma, **diag})
    return records


def summarize(long_df):
    """One row per (dataset, group) with gate values."""
    rows = []
    for (ds, n), d in long_df.groupby(["dataset", "group"]):
        l = d[(d.model == "linear") & (d.t_min.isin(WIN))]
        ref = d[(d.model == "linear") & (d.t_min == REF_T_MIN)]
        if ref.empty or l.beta.isna().all():
            rows.append({"dataset": ds, "group": n, "beta_l": np.nan})
            continue
        r = ref.iloc[0]
        rows.append({"dataset": ds, "group": n,
                     "beta_l": l.beta.median(),
                     "beta_rng": l.beta.max() - l.beta.min(),
                     "beta_spread_max": l.beta_spread.max(),
                     "A": r.A, "M": r.M, "exc_l": r.bin_excess,
                     "snr": abs(r.A) * 0.5 ** (-r.beta) / r.sigma_noise
                            if r.sigma_noise > 0 else np.nan,
                     "t_max": r.t_max_used,
                     "potential_file": r.potential_file})
    return pd.DataFrame(rows)


def main(root, out_long="master_grid.csv", out_summary="master_gates.csv"):
    """Run both datasets and write unified long and summary tables."""
    meta = select_115(Path(root) / XLSX if (Path(root) / XLSX).is_file() else XLSX)
    records, missing = [], []
    for ds, n, path in iter_files(root, meta):
        recs = run_file(path)
        if recs is None:
            missing.append((ds, n))
            continue
        for r in recs:
            r["dataset"], r["group"] = ds, n
        records.extend(recs)
    long_df = pd.DataFrame(records)
    if long_df.empty:
        print(f"no data loaded; missing: {missing}")
        return None
    long_df.to_csv(out_long, index=False)
    summ = summarize(long_df)
    m = meta.rename(columns={"组数": "group"})
    summ = summ.merge(m.assign(dataset="ca115"), on=["dataset", "group"],
                      how="left")
    summ.to_csv(out_summary, index=False)
    print(f"{len(long_df)} rows -> {out_long}")
    print(f"{len(summ)} groups -> {out_summary}")
    if missing:
        print(f"missing: {missing}")
    return summ


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")