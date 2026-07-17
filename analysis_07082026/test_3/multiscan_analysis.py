"""Multi-scan-rate analysis: 20 mV/s pair gets blank-subtracted reversibility; all rates get anodic Randles-Sevcik."""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analysis_07082026.test_3.cv_core as cc

# scan_rate(mV/s) -> (sample Cv folder, blank Cv folder or None)
PAIRS = {20: ("Cv3", "Cv1"), 50: ("Cv4", None), 100: ("Cv5", None),
         200: ("Cv6", None), 500: ("Cv7", None)}
E_STEP_MV = 5.0
ORDERS = [(0.5, "baseline_0.5"), (0.575, "ca_beta")]
HALFWIN_REF_S = 2.0
DT_REF = 0.05  # 100 mV/s reference; HALFWIN scales with dt


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
    out = {}
    for k in keys:
        E, i_s, t = sample[k]
        _, i_b, _ = blank[k]
        if len(i_s) != len(i_b):
            raise ValueError(f"scan {k}: point counts differ ({len(i_s)} vs {len(i_b)})")
        out[k] = (E, i_s - i_b, t)
    return out


def process(iso_dir, rate, sample_dir, blank_dir):
    """One scan rate: transform, nominate, fit; return peak rows for both orders."""
    print(f"  {rate} mV/s ({sample_dir}) ...", flush=True)
    sample = load_cv(Path(iso_dir) / sample_dir / "data.csv")
    if blank_dir is not None:
        scans = subtract(sample, load_cv(Path(iso_dir) / blank_dir / "data.csv"))
        bg = blank_dir
    else:
        scans = sample
        bg = "none"
    keys = sorted(scans)
    i_scans = [scans[k][1] for k in keys]
    E_all = np.concatenate([scans[k][0] for k in keys])
    dt = E_STEP_MV / rate / 1000.0 * 1000.0  # = e_step_mv/rate in seconds per point
    dt = E_STEP_MV / rate
    halfwin = min(HALFWIN_REF_S * (dt / DT_REF), 3.0)
    cc.HALFWIN_S = halfwin  # scale window with scan rate
    rows = []
    for order, order_src in ORDERS:
        for mode in ("continuous",):
            sd, slices, guard = cc.transform(i_scans, dt, order, mode)
            t_all = np.arange(len(sd)) * dt
            cands = cc.nominate(sd, slices, guard, E_all)
            import pandas as pd
            print(pd.DataFrame(cands)[["scan","idx","E","polarity","prom"]].sort_values("E").to_string())
            cls = list(cc.clusters(cands, dt))

            def in_fc_window(cl):
                Es = E_all[[c["idx"] for c in cl]]
                return bool(np.any((Es > 0.10) & (Es < 0.30)))

            cls = [cl for cl in cls if in_fc_window(cl)]
            print(f"    {order_src}: {len(cls)} clusters in Fc window", flush=True)
            for ci, cl in enumerate(cls):
                cl = sorted(cl, key=lambda c: -c["prom"])[:4]
                print(f"      cluster {ci+1}/{len(cls)} ({len(cl)} cands) ...", flush=True)
                comps, status, margin = cc.fit_cluster(cl, sd, t_all, slices, dt,
                                                        "frozen", mode, order)
                for j, c in enumerate(comps):
                    rows.append({"rate": rate, "sample": sample_dir, "bg": bg,
                                 "order": order, "order_src": order_src,
                                 "halfwin_s": round(halfwin, 3), "dt": dt,
                                 "scan": c["scan"], "cluster": ci, "comp": j,
                                 "polarity": c["polarity"], "pon": c["pon"],
                                 "fwhm": c["fwhm"], "ip": c["ip"], "status": status,
                                 "Ep": float(np.interp(c["center"], t_all, E_all))})
    return rows


def anodic_families(peaks):
    """Median anodic ip per (rate, order): the dominant positive-polarity peak family."""
    df = peaks[(peaks.polarity == 1)].copy()
    out = []
    for (rate, order_src), g in df.groupby(["rate", "order_src"]):
        # dominant family = highest median pon among Ep-clusters; here take strongest peak per scan then median
        g = g.sort_values("pon", ascending=False)
        top_ep = g.iloc[0]["Ep"]
        fam = g[(g.Ep - top_ep).abs() <= 0.02]
        out.append({"rate": rate, "order_src": order_src,
                    "Epa_med": float(fam.Ep.median()),
                    "ipa_med": float(fam.ip.median()),
                    "ipa_mad": float(np.median(np.abs(fam.ip - fam.ip.median()))),
                    "fwhm_med": float(fam.fwhm.median()),
                    "n": len(fam)})
    return pd.DataFrame(out)


def randles(anod):
    """Regress ipa vs sqrt(v) per order; report slope, intercept, R^2."""
    out = []
    for order_src, g in anod.groupby("order_src"):
        g = g.sort_values("rate")
        x = np.sqrt(g.rate.values)
        y = np.abs(g.ipa_med.values)
        n = len(x)
        A = np.vstack([x, np.ones(n)]).T
        (slope, icpt), *_ = np.linalg.lstsq(A, y, rcond=None)
        yhat = slope * x + icpt
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        # intercept as fraction of the mean signal: closeness to origin
        out.append({"order_src": order_src, "slope": slope, "intercept": icpt,
                    "intercept_frac": icpt / y.mean() if y.mean() else np.nan,
                    "r2": r2, "n_rates": n})
    return pd.DataFrame(out)


def main(iso_dir, out_peaks="ms_peaks.csv", out_rs="ms_randles.csv"):
    """Run all rates; write peak table, anodic families, Randles-Sevcik regression."""
    rows = []
    for rate, (sdir, bdir) in PAIRS.items():
        if not (Path(iso_dir) / sdir / "data.csv").is_file():
            print(f"skip {rate} mV/s: {sdir} missing")
            continue
        rows += process(iso_dir, rate, sdir, bdir)
    peaks = pd.DataFrame(rows)
    peaks.to_csv(out_peaks, index=False)
    anod = anodic_families(peaks)
    rs = randles(anod)
    anod.to_csv("ms_anodic.csv", index=False)
    rs.to_csv(out_rs, index=False)
    print(f"{len(peaks)} peaks -> {out_peaks}")
    print("\n=== anodic ipa by rate & order ===")
    print(anod.sort_values(["order_src", "rate"]).to_string(index=False))
    print("\n=== Randles-Sevcik (ipa vs sqrt v) ===")
    print(rs.to_string(index=False))
    # 20 mV/s reversibility (only rate with blank)
    rev = peaks[(peaks.rate == 20)]
    print("\n=== 20 mV/s peaks (blank-subtracted) ===")
    for order_src in ["baseline_0.5", "ca_beta"]:
        g = rev[rev.order_src == order_src]
        pos = g[g.polarity == 1].sort_values("pon", ascending=False)
        neg = g[g.polarity == -1].sort_values("pon", ascending=False)
        if not pos.empty and not neg.empty:
            ipa, ipc = pos.iloc[0].ip, neg.iloc[0].ip
            print(f"{order_src}: Epa={pos.iloc[0].Ep:.3f} Epc={neg.iloc[0].Ep:.3f} "
                  f"ip_ratio={abs(ipc/ipa):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_iso_5")