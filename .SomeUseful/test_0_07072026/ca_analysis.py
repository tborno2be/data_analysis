"""Quick-look CA fit for test_iso: fits the newest (or given) Ca folder and saves an annotated figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis import load_ca, fit_ca_beta

ROOT = Path(__file__).resolve().parent
ISO = ROOT / "test_iso_2"
# GRAPH = ROOT / "graph"


def newest_ca(iso: Path) -> Path:
    """Latest Ca<n> folder by number."""
    cas = [(int(p.name[2:]), p) for p in iso.iterdir()
           if p.is_dir() and p.name.startswith("Ca") and p.name[2:].isdigit()]
    if not cas:
        raise SystemExit("No Ca folders in test_iso.")
    return max(cas)[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label", help="sample or blank (free text, goes into title and filename)")
    ap.add_argument("wait", help="wait time before this CA, e.g. 3min / 30s / fresh (free text)")
    ap.add_argument("--ca", type=int, default=None, help="Ca folder number (default: newest)")
    args = ap.parse_args()

    folder = ISO / f"Ca{args.ca}" if args.ca else newest_ca(ISO)
    potential_v, t, i = load_ca(folder / "data.csv")
    duration_min = (t[-1] - t[0]) / 60.0

    fit = fit_ca_beta(t, i)

    # GRAPH.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, i * 1e6, ".", ms=3, c="tab:gray", label="data")
    if not np.isnan(fit["beta"]):
        tt = np.linspace(max(fit["t_min_used"], t[0]), t[-1], 500)
        ax.plot(tt, (fit["a"] * tt ** (-fit["beta"]) + fit["c"]) * 1e6, c="tab:red",
                label=f"fit: beta={fit['beta']:.3f}, c={fit['c']:.2e}, r2={fit['r2']:.4f}")
        ax.axvline(fit["t_min_used"], ls=":", c="k", lw=0.8)
    else:
        ax.set_facecolor("#fff5f5")
    ax.set_xlabel("t / s")
    ax.set_ylabel("i / uA")
    title = (f"{args.label} | {potential_v:.2f} V | wait {args.wait} | "
             f"ran {duration_min:.1f} min | {folder.name}")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)

    # out = GRAPH / f"ca_{folder.name}_{args.label}_{args.wait}.png"
    out = folder / f"fit_{args.label}_{args.wait}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{folder.name}: beta={fit['beta']:.4f} c={fit['c']:.3e} r2={fit['r2']:.4f} "
          f"plateau={fit['plateau_found']} t_min={fit['t_min_used']:.2f}s")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()