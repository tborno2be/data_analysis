"""Scan the fractional-derivative order and score peak quality against theory anchors."""
import csv
import sys
from pathlib import Path

import numpy as np

from analysis import (FC_E0, group_metadata, load_cv, mean_blank_scans,
                      background_plan, semideriv_full, _branch_masks,
                      _fit_pearson_peak)
from korobka.cyclic_voltametry.EmStat4X import reconstruct_time

ORDERS = [0.40, 0.45, 0.50, 0.525, 0.55, 0.575, 0.60, 0.65, 0.70]
DT = 0.05


def score_scan(E, i_s, i_b, order, tag):
    """One scan at one order: both branches, Pearson metrics, Ehalf vs FC_E0."""
    n = len(i_s) if i_b is None else min(len(i_s), len(i_b))
    if n < 30:
        return []
    E = E[:n]
    diff = i_s[:n] if i_b is None else i_s[:n] - i_b[:n]
    t = np.asarray(reconstruct_time(n, DT), dtype=float)
    sd = semideriv_full(diff, DT, order)
    fwd, rev = _branch_masks(E)
    an = _fit_pearson_peak(t[fwd], sd[fwd], E[fwd], DT, anodic=True, order=order)
    ca = _fit_pearson_peak(t[rev], sd[rev], E[rev], DT, anodic=False, order=order)
    rows = []
    for branch, res in (("anodic", an), ("cathodic", ca)):
        if res is None:
            continue
        rows.append({**tag, "order": order, "branch": branch, **res,
                     "abs_skew": abs(res["skew"])})
    if an and ca:
        ehalf = (an["Ep"] + ca["Ep"]) / 2
        for r in rows:
            r["Ehalf"] = ehalf
            r["Ehalf_err"] = ehalf - FC_E0
            r["ip_ratio"] = abs(ca["ip"] / an["ip"]) if an["ip"] else np.nan
    return rows


def run_115(cv_root, rows):
    """T-day samples against the T_20 mean blank, all scans, all orders."""
    if not (Path(cv_root) / "96" / "data.csv").is_file():
        print(f"WARNING: no 115-set CV data under {cv_root}")
        return
    meta = group_metadata()
    plan = next(p for p in background_plan(meta) if p["tag"] == "T_20")
    blank_mean = mean_blank_scans(plan["blanks"], cv_root)
    for smp in plan["samples"]:
        path = Path(cv_root) / str(smp) / "data.csv"
        if not path.is_file():
            continue
        cv = load_cv(path)
        for k in sorted(cv["scans"]):
            if k not in blank_mean:
                continue
            E, i_s, _ = cv["scans"][k]
            E_b, i_b = blank_mean[k]
            for g in ORDERS:
                rows.extend(score_scan(E, i_s, i_b, g,
                                       {"dataset": "ca115_T", "sample": smp,
                                        "scan": k, "no_blank": False}))


def run_testiso(tdir, rows):
    """Cv1/Cv2 without blank subtraction, flagged."""
    if not (Path(tdir) / "Cv1" / "data.csv").is_file():
        print(f"WARNING: no Cv files under {tdir} "
              f"(expected {Path(tdir) / 'Cv1' / 'data.csv'})")
        return
    for name in ("Cv1", "Cv2"):
        path = Path(tdir) / name / "data.csv"
        if not path.is_file():
            print(f"WARNING: missing {path}")
            continue
        cv = load_cv(path)
        for k in sorted(cv["scans"]):
            E, i_s, _ = cv["scans"][k]
            for g in ORDERS:
                rows.extend(score_scan(E, i_s, None, g,
                                       {"dataset": "test_iso", "sample": name,
                                        "scan": k, "no_blank": True}))


def main(cv_root, testiso_dir, out_csv="order_scan.csv"):
    """Run both datasets across the order grid and write the long table."""
    rows = []
    run_115(cv_root, rows)
    run_testiso(testiso_dir, rows)
    if not rows:
        print("no rows produced; check both paths")
        return
    counts = {}
    for r in rows:
        counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
    fields = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out_csv}; per dataset: {counts}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python order_scan.py <cv_115_root> <testiso_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])