"""Six diagnostic plots from the analysis outputs. Run after build_diagnostic_tables + export_pointwise.

Expects in --data dir:
  table4_sample_readings.csv, table5_blank_peaks.csv,
  blank_replicate_pointwise.csv, baseline_slope.csv
Writes PNGs to --out dir. Groups are (polished, day); never mixed across groups.
"""

from __future__ import annotations
import argparse
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _group_label(pol, day):
    return f"{'polished' if str(pol)=='True' else 'unpolished'}·{day}"


def _sample_group(plan):
    """Collapse plan tag to (polished, day) sample group."""
    if plan.startswith("Y_unpolished"):
        return "unpolished·Y"
    if plan.startswith("Y_polished"):
        return "polished·Y"
    if plan.startswith("T_"):
        return "polished·T"
    return plan


# 1. blank curve replicate SD vs E, one line per group
def plot_replicate_sd(data, out):
    rows = _read(data / "blank_replicate_pointwise.csv")
    g = defaultdict(list)
    for r in rows:
        g[_group_label(r["polished"], r["day"])].append((_f(r["E"]), _f(r["current_sd"])))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, pts in sorted(g.items()):
        pts.sort()
        E = np.array([p[0] for p in pts])
        sd = np.array([p[1] for p in pts]) * 1e9
        ax.plot(E, sd, lw=1.3, label=f"{label} (n_repl={_read_nrepl(rows, label)})")
    ax.set_xlabel("Potential E (V)")
    ax.set_ylabel("Replicate SD of current (nA)")
    ax.set_title("Blank curve replicate SD vs potential (per group)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "1_blank_replicate_sd.png", dpi=150)
    plt.close(fig)


def _read_nrepl(rows, label):
    for r in rows:
        if _group_label(r["polished"], r["day"]) == label:
            return r["n_replicates"]
    return "?"


# 2. blank RMS difference vs baseline, by group (needs resid_rms in table5)
def plot_rms_difference(data, out):
    rows = _read(data / "table5_blank_peaks.csv")
    if not rows or "resid_rms" not in rows[0]:
        _placeholder(out / "2_blank_rms_difference.png",
                     "resid_rms not in table5\n(re-run with the updated analysis.py)")
        return
    # one value per (blank, scan); dedup by blank+scan
    seen = {}
    for r in rows:
        key = (r["blank"], r["scan"])
        seen[key] = (_group_label(r["polished"], r["day"]), _f(r.get("resid_rms")))
    g = defaultdict(list)
    for label, v in seen.values():
        if not np.isnan(v):
            g[label].append(v * 1e9)
    _bar_mean_sd(g, out / "2_blank_rms_difference.png",
                 "Blank RMS difference vs baseline (nA)", "resid_rms (nA)")


# 3. blank fake peak_over_noise (false-positive floor), by group, vs Fc real-peak level
def plot_fake_pon(data, out):
    brows = _read(data / "table5_blank_peaks.csv")
    g = defaultdict(list)
    for r in brows:
        if r.get("branch") not in (None, "", "none"):
            g[_group_label(r["polished"], r["day"])].append(_f(r["peak_over_noise"]))
    # Fc real-peak pon from table4 (anodic) for reference line
    srows = _read(data / "table4_sample_readings.csv")
    fc_pon = [_f(r["pearson_a_peak_over_noise"]) for r in srows]
    fc_pon = [x for x in fc_pon if not np.isnan(x)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = sorted(g)
    data_arr = [g[l] for l in labels]
    if any(data_arr):
        ax.boxplot([d if d else [np.nan] for d in data_arr], tick_labels=labels, showfliers=True)
    if fc_pon:
        ax.axhline(np.median(fc_pon), color="crimson", ls="--", lw=1.5,
                   label=f"Fc real-peak median pon = {np.median(fc_pon):.1f}")
    ax.set_ylabel("peak_over_noise")
    ax.set_title("Blank fake peak_over_noise (false-positive floor) vs Fc real peak")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "3_blank_fake_pon.png", dpi=150)
    plt.close(fig)


# 4. baseline slope variability by group
def plot_slope_variability(data, out):
    rows = _read(data / "baseline_slope.csv")
    g = defaultdict(list)
    for r in rows:
        g[_group_label(r["polished"], r["day"])].append(_f(r["slope"]) * 1e6)
    _bar_mean_sd(g, out / "4_baseline_slope_variability.png",
                 "Baseline slope variability by group\n(background window, current vs potential)",
                 "slope (µA/V)")


# 5. sample peak position stability: Epa / Epc / E_half mean±SD by group
def plot_peak_position(data, out):
    rows = _read(data / "table4_sample_readings.csv")
    metrics = {"Epa": "pearson_a_Ep", "Epc": "pearson_c_Ep", "E_half": "pearson_Ehalf"}
    g = {m: defaultdict(list) for m in metrics}
    for r in rows:
        grp = _sample_group(r["plan"])
        for m, col in metrics.items():
            v = _f(r.get(col))
            if not np.isnan(v):
                g[m][grp].append(v)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    groups = sorted({grp for m in metrics for grp in g[m]})
    x = np.arange(len(groups))
    width = 0.25
    for k, m in enumerate(metrics):
        means = [np.mean(g[m][grp]) if g[m][grp] else np.nan for grp in groups]
        sds = [np.std(g[m][grp], ddof=1) if len(g[m][grp]) > 1 else 0 for grp in groups]
        ax.errorbar(x + (k - 1) * width, means, yerr=sds, fmt="o", capsize=4, label=m)
    ax.axhline(0.19, color="gray", ls=":", lw=1, label="Fc E0 = 0.19V")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Potential (V)")
    ax.set_title("Sample peak position stability (mean ± SD, per group)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "5_sample_peak_position.png", dpi=150)
    plt.close(fig)


# 6. Pearson IV fit quality R^2 by group
def plot_r2(data, out):
    rows = _read(data / "table4_sample_readings.csv")
    if not rows or "pearson_a_r2" not in rows[0]:
        _placeholder(out / "6_pearson_r2.png",
                     "pearson_a_r2 not in table4\n(re-run with the updated analysis.py)")
        return
    g = defaultdict(list)
    for r in rows:
        v = _f(r.get("pearson_a_r2"))
        if not np.isnan(v):
            g[_sample_group(r["plan"])].append(v)
    _bar_mean_sd(g, out / "6_pearson_r2.png",
                 "Pearson IV fit quality R² (anodic, per group)", "R²", ylim=(0, 1.02))


# ---- helpers ----
def _bar_mean_sd(g, path, title, ylabel, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = sorted(g)
    means = [np.mean(g[l]) if g[l] else np.nan for l in labels]
    sds = [np.std(g[l], ddof=1) if len(g[l]) > 1 else 0 for l in labels]
    ax.bar(labels, means, yerr=sds, capsize=5, color="#4a7", alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _placeholder(path, msg):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12, color="crimson")
    ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with the CSV outputs")
    ap.add_argument("--out", required=True, help="dir for PNGs")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_replicate_sd(data, out)
    plot_rms_difference(data, out)
    plot_fake_pon(data, out)
    plot_slope_variability(data, out)
    plot_peak_position(data, out)
    plot_r2(data, out)
    print("wrote 6 plots to", out)


if __name__ == "__main__":
    main()