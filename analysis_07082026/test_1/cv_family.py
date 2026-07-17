"""Aggregate cv_peaks.csv into cross-scan peak families per configuration."""

import csv
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

E_TOL_V = 0.02
N_SCANS = 5
CONFIG = ["file", "solution", "order_src", "transform_mode", "fit_line", "polarity"]


def families(df, e_tol=E_TOL_V, n_scans=N_SCANS):
    """Within each config, cluster peaks by Ep; one row per family."""
    pk = df[df["comp"] >= 0].dropna(subset=["Ep"]).copy()
    out = []
    for key, grp in pk.groupby(CONFIG):
        grp = grp.sort_values("Ep")
        fam, cur = [], [grp.iloc[0]]
        for _, r in grp.iloc[1:].iterrows():
            if r["Ep"] - cur[-1]["Ep"] <= e_tol:
                cur.append(r)
            else:
                fam.append(cur)
                cur = [r]
        fam.append(cur)
        for members in fam:
            eps = np.array([m["Ep"] for m in members])
            pons = np.array([m["pon"] for m in members])
            fwhms = np.array([m["fwhm"] for m in members])
            ips = np.array([m["ip"] for m in members])
            scans = {int(m["scan"]) for m in members}
            row = dict(zip(CONFIG, key))
            row.update({
                "Ep_med": float(np.median(eps)),
                "Ep_mad": float(np.median(np.abs(eps - np.median(eps)))),
                "pon_med": float(np.median(pons)),
                "fwhm_med": float(np.median(fwhms)),
                "ip_med": float(np.median(ips)),
                "detection_rate": len(scans) / n_scans,
                "n_hits": len(members),
            })
            out.append(row)
    return pd.DataFrame(out)


def ip_ratios(fam):
    """Pair the dominant +/- family per (file,order,mode,line); |ipc/ipa|."""
    grp_keys = ["file", "solution", "order_src", "transform_mode", "fit_line"]
    out = []
    for key, g in fam.groupby(grp_keys):
        pos = g[g.polarity == 1].sort_values("pon_med", ascending=False)
        neg = g[g.polarity == -1].sort_values("pon_med", ascending=False)
        if pos.empty or neg.empty:
            continue
        a, c = pos.iloc[0], neg.iloc[0]
        row = dict(zip(grp_keys, key))
        row.update({
            "Epa": a["Ep_med"], "Epc": c["Ep_med"],
            "E_half": (a["Ep_med"] + c["Ep_med"]) / 2,
            "ipa": a["ip_med"], "ipc": c["ip_med"],
            "ip_ratio": abs(c["ip_med"] / a["ip_med"]) if a["ip_med"] else np.nan,
            "dr_min": min(a["detection_rate"], c["detection_rate"]),
        })
        out.append(row)
    return pd.DataFrame(out)


def main(in_csv="cv_peaks.csv", out_csv="cv_families.csv", out_ratio="cv_ip_ratios.csv"):
    """Read peaks, aggregate families and ip ratios, write both."""
    df = pd.read_csv(in_csv)
    fam = families(df)
    fam = fam.sort_values(["file", "polarity", "Ep_med"]).reset_index(drop=True)
    fam.to_csv(out_csv, index=False)
    rat = ip_ratios(fam)
    rat = rat.sort_values(["file", "order_src", "transform_mode", "fit_line"]).reset_index(drop=True)
    rat.to_csv(out_ratio, index=False)
    print(f"{len(df[df['comp'] >= 0])} peaks -> {len(fam)} families -> {out_csv}")
    print(f"{len(rat)} reversible pairs -> {out_ratio}")
    return fam, rat


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cv_peaks.csv",
         sys.argv[2] if len(sys.argv) > 2 else "cv_families.csv")