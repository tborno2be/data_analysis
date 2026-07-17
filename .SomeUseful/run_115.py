"""Run the CA quality pipeline on filtered groups from the 115-group dataset."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_ca import (T_MINS, REF_T_MIN, load_ca, fit_one, adaptive_tmax,
                         resid_diag)

XLSX = "Book1.xlsx"
SHEET = "Sheet1"
PATH_PATTERN = "{root}/{n}/data.csv"
WIN = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
GATE_PLATEAU = 0.02
GATE_EXCESS = 3.0
GATE_A_DEV = 0.05


def select_groups(xlsx):
    """Return metadata rows for Sample/Blank groups without electrode disconnect."""
    meta = pd.read_excel(xlsx, sheet_name=SHEET)
    return meta[meta["物质"].isin(["Sample", "Blank"])
                & (meta["断电极？"] == "X")].reset_index(drop=True)


def run_group(n, root):
    """Run the fitting grid on one group; returns list of record dicts or None."""
    path = Path(PATH_PATTERN.format(root=root, n=n))
    if not path.is_file():
        return None
    potential_file, t, i = load_ca(path)
    t_max, sigma = adaptive_tmax(t, i)
    records = []
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
                          "rho1_bin": np.nan, "bin_excess": np.nan})
            records.append({"组数": n, "potential_file": potential_file,
                            "t_min": t_min, "model": model, **res,
                            "n_points": len(ts), "t_max_used": t_max,
                            "sigma_noise": sigma, **diag})
    return records


def summarize(long_df, meta):
    """Build one-row-per-group gate table from the long results."""
    rows = []
    for n, d in long_df.groupby("组数"):
        l = d[(d.model == "linear") & (d.t_min.isin(WIN))]
        ref = d[(d.model == "linear") & (d.t_min == REF_T_MIN)]
        if ref.empty or l.beta.isna().all():
            rows.append({"组数": n, "beta_l": np.nan})
            continue
        rows.append({"组数": n,
                     "beta_l": l.beta.median(),
                     "beta_rng": l.beta.max() - l.beta.min(),
                     "A": ref.A.iloc[0], "M": ref.M.iloc[0],
                     "exc_l": ref.bin_excess.iloc[0],
                     "t_max": ref.t_max_used.iloc[0],
                     "potential_file": ref.potential_file.iloc[0]})
    summ = pd.DataFrame(rows).merge(meta, on="组数", how="left")
    cohort = summ.groupby(["物质", "CA potential", "打磨情况"])["A"].transform("median")
    summ["A_dev"] = (summ.A - cohort).abs() / cohort.abs()
    summ["gate_plateau"] = summ.beta_rng <= GATE_PLATEAU
    summ["gate_excess"] = summ.exc_l <= GATE_EXCESS
    summ["gate_A"] = summ.A_dev <= GATE_A_DEV
    summ["pass_all"] = summ.gate_plateau & summ.gate_excess & summ.gate_A
    summ["potential_match"] = np.isclose(
        pd.to_numeric(summ.potential_file, errors="coerce"),
        summ["CA potential"], atol=1e-6)
    return summ


def main(root, out_long="ca115_grid.csv", out_summary="ca115_gates.csv"):
    """Run all filtered groups and write long and per-group summary CSVs."""
    meta = select_groups(XLSX)
    records, missing = [], []
    for n in meta["组数"].dropna().astype(int):
        recs = run_group(n, root)
        if recs is None:
            missing.append(n)
        else:
            records.extend(recs)
    long_df = pd.DataFrame(records)
    if long_df.empty:
        print(f"no data loaded; missing 组数: {missing}")
        return None
    long_df.to_csv(out_long, index=False)
    summ = summarize(long_df, meta)
    summ.to_csv(out_summary, index=False)
    print(f"{len(long_df)} rows -> {out_long}")
    print(f"{len(summ)} groups -> {out_summary}; pass_all: {int(summ.pass_all.sum())}")
    if missing:
        print(f"missing data files for 组数: {missing}")
    return summ


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
    