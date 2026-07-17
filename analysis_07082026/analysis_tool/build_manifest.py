"""Read a test's session logs, pair each measurement block with its background, write result/manifest.csv.

Rules: a block = consecutive same-solution CV/CA rows. The first blank of the run is the baseline.
Each sample uses the most recent preceding blank as its background (the current-domain subtraction).
Blanks are backed against the baseline. Multiple session_*.csv are merged in time order (test_iso_3
has two sessions that belong to the same run)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def read_sessions(iso_dir):
    """Merge all session_*.csv rows in time order, dropping repeated header lines."""
    rows = []
    for p in sorted(Path(iso_dir).glob("session_*.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            rows += [r for r in csv.DictReader(f)
                     if r.get("action") and r.get("step") != "step"]
    rows.sort(key=lambda r: r["t_start"])
    return rows


def blocks(rows):
    """Group consecutive same-solution CV/CA rows into measurement blocks."""
    out = []
    for r in rows:
        if r["action"] not in ("CV", "CA") or not r["out_dir"]:
            continue
        sol = r["solution"]
        if not out or out[-1]["solution"] != sol or r["action"] in out[-1]:
            out.append({"solution": sol, "t_start": r["t_start"]})
        out[-1][r["action"]] = r["out_dir"]
    return out


def pair(blks):
    """Attach background: baseline=first blank; blanks->baseline; samples->preceding blank."""
    baseline, last_blank, out = None, None, []
    for b in blks:
        if b["solution"].startswith("blank"):
            if baseline is None:
                baseline = b
                b["role"] = "baseline"
            else:
                b["role"] = "blank"
            b["bg"] = baseline
            last_blank = b
        else:
            if last_blank is None:
                continue
            b["role"] = "sample"
            b["bg"] = last_blank
        out.append(b)
    return out


def manifest(iso_dir):
    """Rows: role, cv, ca, bg_cv, bg_ca, t_start."""
    paired = pair(blocks(read_sessions(iso_dir)))
    rows = []
    for b in paired:
        bg = b["bg"]
        rows.append({"role": b["role"], "t_start": b["t_start"],
                     "cv": b.get("CV", ""), "ca": b.get("CA", ""),
                     "bg_cv": bg.get("CV", ""), "bg_ca": bg.get("CA", "")})
    return rows


def main(iso_dir):
    """Write result/manifest.csv under the iso dir."""
    iso = Path(iso_dir)
    rows = manifest(iso)
    if not rows:
        print(f"no blocks found in {iso_dir}")
        return
    out_dir = iso / "result"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "manifest.csv"
    fields = ["role", "t_start", "cv", "ca", "bg_cv", "bg_ca"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} blocks -> {out}")
    for r in rows:
        print(f"  {r['role']:8} cv={r['cv']:4} ca={r['ca']:4} "
              f"bg_cv={r['bg_cv']:4} bg_ca={r['bg_ca']:4}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python build_manifest.py <iso_dir>  (e.g. test_1/test_iso_3)")
        sys.exit(1)
    main(sys.argv[1])