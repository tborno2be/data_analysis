"""Driver: pair session CVs with their blank backgrounds and run the 2x3 core ablation."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

import analysis_07082026.test_3.cv_core as cc

ORDER = 0.5  # CA-certified beta plugs in here
MODES = ("continuous", "per_scan")


def read_sessions(iso_dir):
    """Merge session_*.csv rows in time order, tolerating repeated header lines."""
    rows = []
    for p in sorted(Path(iso_dir).glob("session_*.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            rows += [r for r in csv.DictReader(f)
                     if r.get("action") and r["step"] != "step"]
    rows.sort(key=lambda r: r["t_start"])
    return rows


def blocks(rows):
    """Consecutive same-solution CV/CA rows grouped into measurement blocks."""
    out = []
    for r in rows:
        if r["action"] not in ("CV", "CA") or not r["out_dir"]:
            continue
        if not out or out[-1]["solution"] != r["solution"] or r["action"] in out[-1]:
            out.append({"solution": r["solution"]})
        out[-1][r["action"]] = r["out_dir"]
    return out


def pair(blks):
    """Attach background block: first blank for blanks, preceding blank for samples."""
    baseline, last_blank, out = None, None, []
    for b in blks:
        if b["solution"].startswith("blank"):
            baseline = baseline or b
            b["bg"] = baseline
            last_blank = b
        elif last_blank:
            b["bg"] = last_blank
        else:
            continue
        out.append(b)
    return out


def cv_jobs(rows):
    """(cv_dir, solution, bg_cv_dir) from paired blocks."""
    return [(b["CV"], b["solution"], b["bg"]["CV"]) for b in pair(blocks(rows))
            if "CV" in b and "CV" in b["bg"]]


def ca_jobs(rows):
    """(ca_dir, solution, bg_ca_dir, own_cv_dir) from paired blocks, for the beta layer."""
    return [(b["CA"], b["solution"], b["bg"].get("CA", ""), b.get("CV", ""))
            for b in pair(blocks(rows)) if "CA" in b]


def load_cv(path):
    """Read a CV data.csv into {k: (E, i, t)}, equilibration rows dropped."""
    scans = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            scan, e, i, t = line.rstrip("\n").split(",")
            if scan == "":
                continue
            scans.setdefault(int(scan), []).append((float(e), float(i), float(t) if t else np.nan))
    return {k: tuple(np.array(c) for c in zip(*rows)) for k, rows in scans.items()}


def subtract(sample, blank):
    """Per-scan sample-minus-blank; raises on mismatch."""
    keys = sorted(set(sample) & set(blank))
    if not keys:
        raise ValueError("no common scans")
    out = {}
    for k in keys:
        E, i_s, t = sample[k]
        _, i_b, _ = blank[k]
        if len(i_s) != len(i_b):
            raise ValueError(f"scan {k}: point counts differ")
        out[k] = (E, i_s - i_b, t)
    return out


def analyze(iso_dir, cv_dir, sol, bg_dir, orders):
    """One paired file through every (order, mode, line); orders is [(value, label), ...]."""
    resid = subtract(load_cv(Path(iso_dir) / cv_dir / "data.csv"),
                     load_cv(Path(iso_dir) / bg_dir / "data.csv"))
    keys = sorted(resid)
    i_scans = [resid[k][1] for k in keys]
    E_all = np.concatenate([resid[k][0] for k in keys])
    t = resid[keys[0]][2]
    dt = float(t[1] - t[0]) if np.isfinite(t[1]) else 0.05
    empty = {"scan": -1, "cluster": -1, "comp": -1, "polarity": 0, "Ep": np.nan,
             "fwhm": np.nan, "pon": np.nan, "ip": np.nan, "bic_margin": np.nan}
    rows = []
    for order, order_src in orders:
        base = {"file": cv_dir, "solution": sol, "bg": bg_dir,
                "order": order, "order_src": order_src}
        for mode in MODES:
            sd, slices, guard = cc.transform(i_scans, dt, order, mode)
            t_all = np.arange(len(sd)) * dt
            cands = cc.nominate(sd, slices, guard)
            cls = cc.clusters(cands, dt)
            for line in cc.LINES:
                tag = {**base, "transform_mode": mode, "fit_line": line}
                if not cands:
                    rows.append({**tag, **empty, "status": "no_peak"})
                    continue
                for ci, cl in enumerate(cls):
                    comps, status, margin = cc.fit_cluster(cl, sd, t_all, slices, dt, line, mode, order)
                    if not comps:
                        rows.append({**tag, **empty, "scan": cl[0]["scan"], "cluster": ci,
                                     "bic_margin": margin, "status": status})
                    for j, c in enumerate(comps):
                        rows.append({**tag, "scan": c["scan"], "cluster": ci, "comp": j,
                                     "polarity": c["polarity"],
                                     "Ep": float(np.interp(c["center"], t_all, E_all)),
                                     "fwhm": c["fwhm"], "pon": c["pon"], "ip": c["ip"],
                                     "bic_margin": margin, "status": status})
    return rows


def main(iso_dir, order=ORDER, out_csv="cv_peaks.csv", use_ca_beta=True):
    """Run the ablation over all registered CV pairs; each sample CV uses its CA-certified beta."""
    rows_meta = read_sessions(iso_dir)
    jobs = cv_jobs(rows_meta)
    if not jobs:
        print("no CV jobs found in session logs")
        return
    betas = {}
    if use_ca_beta:
        import analysis_07082026.test_1.ca_fit as ca_fit
        betas = ca_fit.session_betas(iso_dir)
    rows = []
    for cv_dir, sol, bg_dir in jobs:
        orders = [(order, "baseline_0.5")]
        b = betas.get(cv_dir)
        if b is not None and abs(b - order) > 1e-6:
            orders.append((b, "ca_beta"))
        print(f"  {cv_dir} ({sol}) orders={[round(o,3) for o,_ in orders]} ...", flush=True)
        rows += analyze(iso_dir, cv_dir, sol, bg_dir, orders)
    fields = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    used = {cv: round(betas[cv], 3) for cv in betas if cv in {j[0] for j in jobs}}
    print(f"{len(jobs)} files, {len(rows)} rows -> {out_csv}; CA-certified beta: {used or 'none, default 0.5'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_iso_3",
         float(sys.argv[2]) if len(sys.argv) > 2 else ORDER)