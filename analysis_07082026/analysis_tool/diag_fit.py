"""Diagnose one sample CV's fit quality: does the real Fc peak fit cleanly (skew near 0) or does
Pearson fail (skew pinned at +/-2)? Isolates 'background dirty' vs 'Pearson itself broken'.

Usage: python diag_fit.py <iso_dir> <sample_cv> <blank_cv>
  e.g. python diag_fit.py ../test_2/test_iso_4 Cv2 Cv1
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import Project.electrochem.cv_core as cc
import Project.electrochem.cv_peak as cv_peak


def load_cv(path):
    scans = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 3 or p[0] == "":
                continue
            scans.setdefault(int(p[0]), []).append(
                (float(p[1]), float(p[2]), float(p[3]) if len(p) > 3 and p[3] else np.nan))
    return {k: tuple(np.array(c) for c in zip(*rows)) for k, rows in scans.items()}


def main(iso_dir, cv, bg):
    iso = Path(iso_dir)
    sample = load_cv(iso / cv / "data.csv")
    blank = load_cv(iso / bg / "data.csv")
    dt_arr = sample[sorted(sample)[0]][2]
    dt = float(dt_arr[1] - dt_arr[0]) if np.isfinite(dt_arr[1]) else 0.05
    print(f"{cv} minus {bg}, dt={dt}, scans={sorted(sample)}")

    resid = cc.subtract_blank(sample, blank)
    cands = cc.nominate_current(resid, dt)
    print(f"nominate: {len(cands)} candidates")
    for c in cands:
        print(f"  scan{c['scan']} E={c['E_center']:.3f} pol={c['polarity']:+d} "
              f"win=[{c['E_lo']:.3f},{c['E_hi']:.3f}] prom={c['prom']:.2e}")

    sd, slices, guard, t_all, E_all, verts = cc.transform(resid, dt, 0.5, "cross_scan")
    conf = cc.confirm_sd(sd, cands, slices, E_all, guard)
    clusters = cc.cluster(conf)
    print(f"confirm: {len(conf)}, clusters: {len(clusters)} sizes {[len(c) for c in clusters]}")

    pm = cv_peak.PearsonPeak()
    print("\n--- fits (watch skew: ~0 = clean, +/-2 = pinned/failed) ---")
    for cl in clusters:
        fr = cc.fit_in_cluster(cl, sd, slices, t_all, dt, 0.5, "frozen", "cross_scan", pm, verts)
        rp = cc.read_params(fr, 0.5, dt, t_all, E_all)
        for d in rp["detections"]:
            pinned = "  <-- PINNED" if abs(abs(d["shape_diag"]) - 2.0) < 0.01 else ""
            print(f"  pol={d['polarity']:+d} Ep={d['Ep']:.3f} ip={d['ip']:.2e} "
                  f"fwhm={d['fwhm']:.3f} skew={d['shape_diag']:+.3f} pon={d['pon']:.1f} "
                  f"conf={d['confidence']:.3f} status={d['cluster_status']}{pinned}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: python diag_fit.py <iso_dir> <sample_cv> <blank_cv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])