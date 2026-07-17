"""Run CV + CA analysis for one test. Reads result/manifest.csv, subtracts each block's background,
runs the shape-agnostic core over the 16-combo grid (order x model x bg x mode), writes result CSVs.

Usage: python run_analysis.py <iso_dir>      e.g.  python run_analysis.py test_2/test_iso_4
Runs one test at a time (by design). CA analysis runs first to certify each sample's beta."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

import Project.electrochem.cv_core as cc
import Project.electrochem.cv_peak as cv_peak
import ca_fit

MODES = ("cross_scan", "per_scan")
BGS = ("frozen", "joint")
MODELS = {"pearson": cv_peak.PearsonPeak(), "gaussian": cv_peak.GaussianPeak()}
BASE_ORDER = 0.5


def load_cv(path):
    """CV data.csv -> {scan:(E,i,t)}, equilibration rows dropped."""
    scans = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            scan, e, i, t = line.rstrip("\n").split(",")
            if scan == "":
                continue
            scans.setdefault(int(scan), []).append(
                (float(e), float(i), float(t) if t else np.nan))
    return {k: tuple(np.array(c) for c in zip(*rows)) for k, rows in scans.items()}


def _dt(scans):
    t = scans[sorted(scans)[0]][2]
    return float(t[1] - t[0]) if np.isfinite(t[1]) else 0.05


def analyze_cv(iso, cv_dir, bg_cv, role, orders):
    """One CV block over the full grid; returns detection rows tagged with every axis."""
    sample = load_cv(iso / cv_dir / "data.csv")
    blank = load_cv(iso / bg_cv / "data.csv")
    resid = cc.subtract_blank(sample, blank)
    dt = _dt(resid)
    rows = []
    for order, order_src in orders:
        for mode in MODES:
            sd, slices, guard, t_all, E_all, verts = cc.transform(resid, dt, order, mode)
            cands = cc.nominate_current(resid, dt)
            conf = cc.confirm_sd(sd, cands, slices, E_all, guard)
            clusters = cc.cluster(conf)
            for mname, pm in MODELS.items():
                for bg in BGS:
                    for cl in clusters:
                        fr = cc.fit_in_cluster(cl, sd, slices, t_all, dt, order, bg, mode, pm, verts)
                        rp = cc.read_params(fr, order, dt, t_all, E_all)
                        for d in rp["detections"]:
                            rows.append({"cv": cv_dir, "role": role, "bg_cv": bg_cv,
                                         "order": round(order, 4), "order_src": order_src,
                                         "mode": mode, "model": mname, "bg": bg, **d})
    return rows


def main(iso_dir):
    """CA first (certify beta), then CV grid; write result/ca_gates.csv and result/cv_peaks.csv."""
    iso = Path(iso_dir)
    man_path = iso / "result" / "manifest.csv"
    if not man_path.is_file():
        print(f"{man_path} missing; run build_manifest.py first")
        return
    with open(man_path, newline="", encoding="utf-8") as f:
        man = list(csv.DictReader(f))

    print("== CA analysis ==")
    betas = ca_fit.session_betas(iso_dir)   # writes result/ca_gates.csv
    print("certified betas:", {k: round(v, 3) for k, v in betas.items()} or "none")

    print("== CV analysis (16-combo grid per block) ==")
    all_rows = []
    for m in man:
        if m["role"] == "baseline" or not m["cv"]:
            continue                          # baseline subtracts to zero; skip
        orders = [(BASE_ORDER, "baseline_0.5")]
        b = betas.get(m["cv"])
        if b is not None and abs(b - BASE_ORDER) > 1e-6:
            orders.append((b, "ca_beta"))
        print(f"  {m['role']:7} {m['cv']} (bg {m['bg_cv']}) orders={[round(o,3) for o,_ in orders]} ...",
              flush=True)
        all_rows += analyze_cv(iso, m["cv"], m["bg_cv"], m["role"], orders)

    out = iso / "result" / "cv_peaks.csv"
    fields = sorted({k for r in all_rows for k in r})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    print(f"{len(all_rows)} detections -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python run_analysis.py <iso_dir>   e.g.  test_2/test_iso_4")
        sys.exit(1)
    main(sys.argv[1])