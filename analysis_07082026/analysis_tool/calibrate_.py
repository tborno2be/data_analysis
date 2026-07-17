"""Calibrate confidence factor transition points and a peak/no-peak threshold.
No-peak (lower anchor): 40 blank groups (no disconnect), each minus its own polish/day baseline.
Real-peak (upper anchor): multiscan sample CVs (Cv3-Cv6) minus the 20 mV/s blank (Cv1).
Writes result/confidence_calib.csv (per-detection) and prints suggested transition points + threshold.

Usage: python calibrate_confidence.py <iso_dir>   e.g.  test_3/test_iso_5
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

import Project.electrochem.cv_core as cc
import Project.electrochem.cv_peak as cv_peak

DT = 0.05  # 100 mV/s, 5 mV step, from the 115-group time column

# no-peak blank groups (Book1 sheet1: Blank, disconnect=X), grouped by baseline
NOPEAK = {
    6:  [7, 8, 9, 10, 21, 22, 23, 24, 25],   # unpolished -> baseline 6
    26: [27, 28, 29, 30, 31, 32, 33, 34, 35],  # polished (day Y) -> baseline 26
    76: [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
         90, 91, 92, 93, 94, 95],              # polished (day T) -> baseline 76
}
# real-peak multiscan samples -> blank Cv1 (20 mV/s). dt differs per rate; read from time col.
REALPEAK = {"Cv3": "Cv1", "Cv4": "Cv1", "Cv5": "Cv1", "Cv6": "Cv1"}

PM = cv_peak.PearsonPeak()


def load_cv(path):
    """CV data.csv -> {scan:(E,i,t)}, equilibration rows (empty scan) dropped."""
    scans = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3 or parts[0] == "":
                continue
            scan = int(parts[0]); e = float(parts[1]); i = float(parts[2])
            t = float(parts[3]) if len(parts) > 3 and parts[3] else np.nan
            scans.setdefault(scan, []).append((e, i, t))
    return {k: tuple(np.array(c) for c in zip(*rows)) for k, rows in scans.items()}


def dt_of(scans, fallback=DT):
    """dt from the time column if present, else fallback (odd 115-groups lack time)."""
    t = scans[sorted(scans)[0]][2]
    return float(t[1] - t[0]) if np.isfinite(t[1]) else fallback


def run_pair(sample_scans, blank_scans, dt, label, group):
    """Full core over one sample/blank pair (cross_scan, order 0.5, frozen); return detection rows."""
    resid = cc.subtract_blank(sample_scans, blank_scans)
    cands = cc.nominate_current(resid, dt)
    sd, slices, guard, t_all, E_all, verts = cc.transform(resid, dt, 0.5, "cross_scan")
    conf = cc.confirm_sd(sd, cands, slices, E_all, guard)
    clusters = cc.cluster(conf)
    rows = []
    for cl in clusters:
        fr = cc.fit_in_cluster(cl, sd, slices, t_all, dt, 0.5, "frozen", "cross_scan", PM, verts)
        rp = cc.read_params(fr, 0.5, dt, t_all, E_all)
        for d in rp["detections"]:
            rows.append({"label": label, "group": group, **d})
    return rows


def main(iso_dir):
    iso = Path(iso_dir)
    cvroot = iso / "CV"
    rows = []

    # ---- no-peak: blank minus its baseline ----
    for base, members in NOPEAK.items():
        bpath = cvroot / str(base) / "data.csv"
        if not bpath.is_file():
            print(f"skip baseline {base}: missing"); continue
        blank = load_cv(bpath)
        for g in members:
            gpath = cvroot / str(g) / "data.csv"
            if not gpath.is_file():
                continue
            sample = load_cv(gpath)
            try:
                rows += run_pair(sample, blank, DT, "nopeak", g)
            except Exception as e:
                print(f"  nopeak {g}: {e}")

    # ---- real-peak: multiscan sample minus Cv1 ----
    for cv, bg in REALPEAK.items():
        sp, bp = iso / cv / "data.csv", iso / bg / "data.csv"
        if not (sp.is_file() and bp.is_file()):
            print(f"skip {cv}: missing"); continue
        sample, blank = load_cv(sp), load_cv(bp)
        dt = dt_of(sample)
        try:
            rows += run_pair(sample, blank, dt, "realpeak", cv)
        except Exception as e:
            print(f"  realpeak {cv}: {e}")

    if not rows:
        print("no detections produced"); return

    out = iso / "result" / "confidence_calib.csv"
    out.parent.mkdir(exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"\n{len(rows)} detections -> {out}")

    # ---- distributions ----
    real = [r for r in rows if r["label"] == "realpeak"]
    nope = [r for r in rows if r["label"] == "nopeak"]

    def dist(rs, key):
        v = np.array([r[key] for r in rs_ if np.isfinite(r[key])]) if (rs_ := rs) else np.array([])
        if len(v) == 0:
            return "n/a"
        return f"n={len(v)} min={v.min():.3g} p10={np.percentile(v,10):.3g} med={np.median(v):.3g} max={v.max():.3g}"

    print("\n=== real-peak (Fc, should be HIGH confidence) ===")
    for k in ("pon", "fwhm", "shape_diag", "confidence"):
        print(f"  {k:11}: {dist(real, k)}")
    print("=== no-peak (blank, should be LOW confidence) ===")
    for k in ("pon", "fwhm", "shape_diag", "confidence"):
        print(f"  {k:11}: {dist(nope, k)}")

    # ---- threshold gap ----
    rc = np.array([r["confidence"] for r in real if np.isfinite(r["confidence"])])
    nc = np.array([r["confidence"] for r in nope if np.isfinite(r["confidence"])])
    if len(rc) and len(nc):
        real_low = np.percentile(rc, 10)
        nope_high = np.percentile(nc, 90)
        print(f"\nreal-peak conf 10th pct = {real_low:.3f}")
        print(f"no-peak  conf 90th pct = {nope_high:.3f}")
        if real_low > nope_high:
            print(f"GAP EXISTS -> suggested peak threshold = {(real_low+nope_high)/2:.3f}")
        else:
            print("OVERLAP -> factors do not fully separate peak/no-peak; inspect which factor fails")
    # suggested transition points from real Fc
    if len(real):
        pons = np.array([r["pon"] for r in real if np.isfinite(r["pon"])])
        fwhms = np.array([r["fwhm"] for r in real if np.isfinite(r["fwhm"])])
        skews = np.array([abs(r["shape_diag"]) for r in real if np.isfinite(r["shape_diag"])])
        print("\n=== suggested transition points (from real Fc) ===")
        if len(pons): print(f"  pon_lo ~ {np.percentile(pons,5)*0.5:.1f}  (real Fc pon 5th pct = {np.percentile(pons,5):.1f})")
        if len(fwhms): print(f"  fwhm_min ~ {fwhms.min()*0.5:.3f}  (real Fc fwhm min = {fwhms.min():.3f})")
        if len(skews): print(f"  skew_bad ~ {np.percentile(skews,95)*1.5:.2f}  (real Fc |skew| 95th = {np.percentile(skews,95):.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python calibrate_confidence.py <iso_dir>"); sys.exit(1)
    main(sys.argv[1])