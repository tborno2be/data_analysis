#!/usr/bin/env python3
"""
Fit CA beta models for selected groups.

Selection rule from Book1:
  - substance is Blank or Sample
  - disconnected electrode column is X / no / false / 0 / blank
  - Yesterday and Today are both kept

Models:
  normal beta:        i(t) = A * t^(-beta) + C
  beta + linear drift:i(t) = A * t^(-beta) + C + M*t

Output: one row per group x fit_start x model.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ---------- metadata ----------

def norm_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_book(book_path: Path) -> pd.DataFrame:
    meta = pd.read_excel(book_path, sheet_name=0)
    rename = {}
    for col in meta.columns:
        c = str(col).strip()
        if c in ["组数", "group", "Group", "group_id"]:
            rename[col] = "group"
        elif c in ["物质", "substance", "Substance", "sample_type"]:
            rename[col] = "substance"
        elif c.lower().replace(" ", "") in ["capotential", "potential", "ca_potential"]:
            rename[col] = "ca_potential"
        elif c in ["打磨情况", "polishing", "polished", "Polishing"]:
            rename[col] = "polishing"
        elif c in ["断电极？", "断电极?", "disconnected", "electrode_disconnected", "broken_electrode"]:
            rename[col] = "disconnected"
        elif c in ["Yesterday/Today", "day", "Day", "date_block"]:
            rename[col] = "day"
    meta = meta.rename(columns=rename)

    required = ["group", "substance", "ca_potential", "polishing", "disconnected", "day"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Book file is missing required columns after renaming: {missing}. Found: {list(meta.columns)}")

    meta["group"] = pd.to_numeric(meta["group"], errors="coerce").astype("Int64")
    meta["substance_norm"] = meta["substance"].map(lambda x: norm_text(x).lower())
    meta["disconnect_norm"] = meta["disconnected"].map(lambda x: norm_text(x).lower())

    # In your Book1, X = not disconnected, Y = disconnected.
    ok_disconnected_values = {"", "x", "no", "n", "false", "0", "none", "not disconnected"}
    selected = meta[
        meta["substance_norm"].isin(["blank", "sample"]) &
        meta["disconnect_norm"].isin(ok_disconnected_values) &
        meta["group"].notna()
    ].copy()
    selected["group"] = selected["group"].astype(int)
    return selected[["group", "substance", "ca_potential", "polishing", "disconnected", "day"]]


# ---------- CA file discovery ----------

def read_mapping_csv(path: Optional[Path]) -> dict[int, Path]:
    if path is None:
        return {}
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    if "group" not in lower or "file" not in lower:
        raise ValueError("mapping CSV must contain columns: group,file")
    out = {}
    for _, row in df.iterrows():
        g = int(row[lower["group"]])
        out[g] = Path(str(row[lower["file"]])).expanduser()
    return out


def extract_group_from_path(path: Path, valid_groups: set[int]) -> Optional[int]:
    """
    Parse the real group number from a CA file path.

    Your current layout is like:
        CA/100/data.csv
        CA/100/raw.txt

    so the group number is often the parent folder name, not the file name.
    This function therefore checks path parts from nearest parent to filename.
    """
    # 1) Best case for your folder layout: CA/<group>/data.csv
    # Check parent folders from nearest to farthest.
    for part in [path.parent.name, *[p.name for p in path.parents[1:4]]]:
        part_clean = part.strip().lower()
        if re.fullmatch(r"\d{1,3}", part_clean):
            num = int(part_clean)
            if num in valid_groups:
                return num
        m = re.fullmatch(r"(?:group|grp|g|ca|组数|组)[_\- #]*(\d{1,3})", part_clean)
        if m:
            num = int(m.group(1))
            if num in valid_groups:
                return num

    # 2) Fallback: group number in file stem, e.g. group_11.csv, ca_96.csv.
    stem = path.stem.lower()
    strong_patterns = [
        r"(?:^|[_\-\s])(group|grp|g|no|run|measurement|meas|ca|组数|组)\s*[_\-#]*\s*(\d{1,3})(?:$|[_\-\s])",
        r"(?:^|[_\-\s])(blank|sample)\s*[_\-#]*\s*(\d{1,3})(?:$|[_\-\s])",
        r"(?:^|[_\-\s])(\d{1,3})(?:$|[_\-\s])",
    ]
    for pat in strong_patterns:
        for m in re.finditer(pat, stem):
            num = int(m.group(m.lastindex))
            if num in valid_groups:
                before = stem[:m.start()]
                if before.endswith("test"):
                    continue
                return num

    if re.fullmatch(r"\d{1,3}", stem):
        num = int(stem)
        if num in valid_groups:
            return num

    return None

def discover_ca_files(ca_dir: Path, valid_groups: Iterable[int], mapping_csv: Optional[Path] = None) -> tuple[dict[int, Path], list[Path]]:
    valid_groups = set(int(g) for g in valid_groups)
    mapped = read_mapping_csv(mapping_csv)
    group_to_file: dict[int, Path] = {}
    unmapped: list[Path] = []

    # Manual mapping wins.
    for g, p in mapped.items():
        if g in valid_groups:
            group_to_file[g] = p if p.is_absolute() else (ca_dir.parent / p)

    exts = {".csv", ".txt", ".tsv", ".dat", ".xlsx", ".xls"}

    def file_preference(path: Path) -> tuple[int, str]:
        """Lower is better. Prefer parsed numeric CSV files over raw logs."""
        name = path.name.lower()
        stem = path.stem.lower()
        if name == "data.csv":
            return (0, name)
        if stem in {"data", "ca", "result", "results"} and path.suffix.lower() == ".csv":
            return (1, name)
        if path.suffix.lower() == ".csv":
            return (2, name)
        if stem == "raw":
            return (9, name)
        return (5, name)

    candidates: dict[int, list[Path]] = {}
    for p in sorted(ca_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        g = extract_group_from_path(p, valid_groups)
        if g is None:
            unmapped.append(p)
            continue
        candidates.setdefault(g, []).append(p)

    for g, paths in candidates.items():
        if g in group_to_file:
            continue
        paths_sorted = sorted(paths, key=file_preference)
        group_to_file[g] = paths_sorted[0]
        unmapped.extend(paths_sorted[1:])

    return group_to_file, unmapped


# ---------- CA reading ----------

def read_table_flex(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    # Try common separators.
    attempts = [
        dict(sep=None, engine="python"),
        dict(sep=","),
        dict(sep="\t"),
        dict(sep=";"),
        dict(delim_whitespace=True),
    ]
    last_err = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, comment="#", **kwargs)
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last_err = e
    raise ValueError(f"Could not read {path}: {last_err}")


def choose_time_current_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = list(df.columns)
    clean = {c: re.sub(r"[^a-z0-9]+", "", str(c).lower()) for c in cols}

    time_keys = ["time", "times", "timesec", "seconds", "second", "ts", "t"]
    current_keys = ["current", "currenta", "currentua", "currentma", "ia", "iu", "iua", "currenti", "i"]

    time_col = None
    current_col = None
    for c, s in clean.items():
        if time_col is None and (s in time_keys or s.startswith("time")):
            time_col = c
        if current_col is None and (s in current_keys or "current" in s or s in {"i", "ia"}):
            current_col = c

    numeric_cols = []
    for c in cols:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() >= max(5, int(0.2 * len(vals))):
            numeric_cols.append(c)

    if time_col is None:
        # Usually first numeric column is time.
        if not numeric_cols:
            raise ValueError("No numeric columns found for time/current.")
        time_col = numeric_cols[0]

    if current_col is None:
        # Usually last numeric column is current if no label is recognized.
        candidates = [c for c in numeric_cols if c != time_col]
        if not candidates:
            raise ValueError("Could not find a current column.")
        current_col = candidates[-1]

    return str(time_col), str(current_col)


def load_ca_trace(path: Path) -> pd.DataFrame:
    raw = read_table_flex(path)
    time_col, current_col = choose_time_current_columns(raw)
    out = pd.DataFrame({
        "time_s": pd.to_numeric(raw[time_col], errors="coerce"),
        "current": pd.to_numeric(raw[current_col], errors="coerce"),
    }).dropna()
    out = out.sort_values("time_s")
    out = out.drop_duplicates(subset=["time_s"])
    # Shift time to start at 0 if the file has absolute/offset time.
    out["time_s"] = out["time_s"] - float(out["time_s"].min())
    return out.reset_index(drop=True)


# ---------- fitting ----------

def model_beta(t, A, beta, C):
    return A * np.power(t, -beta) + C


def model_beta_drift(t, A, beta, C, M):
    return A * np.power(t, -beta) + C + M * t


def fit_one(trace: pd.DataFrame, start_s: float, model_name: str, min_points: int = 20) -> dict:
    fit = trace[trace["time_s"] >= start_s].copy()
    fit = fit[np.isfinite(fit["time_s"]) & np.isfinite(fit["current"])]
    fit = fit[fit["time_s"] > 0]

    if len(fit) < min_points:
        return {
            "success": False,
            "message": f"too few points after start_s={start_s}: {len(fit)}",
            "n_points": len(fit),
        }

    t = fit["time_s"].to_numpy(dtype=float)
    y = fit["current"].to_numpy(dtype=float)

    # Robust-ish initial guesses.
    c0 = float(np.nanmedian(y[-max(3, len(y)//10):]))
    amp_guess = float(y[0] - c0)
    if abs(amp_guess) < 1e-15:
        amp_guess = float(np.nanstd(y) or 1e-12)

    try:
        if model_name == "beta":
            p0 = [amp_guess, 0.5, c0]
            bounds = ([-np.inf, 0.0, -np.inf], [np.inf, 2.0, np.inf])
            popt, pcov = curve_fit(model_beta, t, y, p0=p0, bounds=bounds, maxfev=50000)
            yhat = model_beta(t, *popt)
            A, beta, C = popt
            M = np.nan
        elif model_name == "beta_linear_drift":
            # Drift guess from last 20% of points.
            tail_n = max(5, len(t) // 5)
            try:
                M0 = float(np.polyfit(t[-tail_n:], y[-tail_n:], 1)[0])
            except Exception:
                M0 = 0.0
            p0 = [amp_guess, 0.5, c0, M0]
            bounds = ([-np.inf, 0.0, -np.inf, -np.inf], [np.inf, 2.0, np.inf, np.inf])
            popt, pcov = curve_fit(model_beta_drift, t, y, p0=p0, bounds=bounds, maxfev=80000)
            yhat = model_beta_drift(t, *popt)
            A, beta, C, M = popt
        else:
            raise ValueError(model_name)

        resid = y - yhat
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        # 1-sigma uncertainty if covariance is usable.
        perr = np.sqrt(np.diag(pcov)) if pcov is not None and np.all(np.isfinite(pcov)) else [np.nan] * len(popt)
        beta_se = float(perr[1]) if len(perr) > 1 else np.nan

        return {
            "success": True,
            "message": "ok",
            "n_points": int(len(fit)),
            "fit_start_s": float(start_s),
            "fit_end_s": float(t.max()),
            "model": model_name,
            "A": float(A),
            "beta": float(beta),
            "beta_se": beta_se,
            "C": float(C),
            "M_linear_drift": float(M) if np.isfinite(M) else np.nan,
            "rmse": rmse,
            "r2": r2,
            "current_first_fit_point": float(y[0]),
            "current_last_fit_point": float(y[-1]),
        }
    except Exception as e:
        return {
            "success": False,
            "message": repr(e),
            "n_points": int(len(fit)),
            "fit_start_s": float(start_s),
            "fit_end_s": float(t.max()) if len(t) else np.nan,
            "model": model_name,
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".", help="Project root containing CA/ and Book1 file")
    ap.add_argument("--book", default="Book1.xlsx", help="Book1 metadata xlsx path, relative to project root unless absolute")
    ap.add_argument("--ca-dir", default="CA", help="CA folder path, relative to project root unless absolute")
    ap.add_argument("--out", default="ca_beta_fits.csv", help="Output CSV path")
    ap.add_argument("--starts", nargs="+", type=float, default=[0.2, 0.5, 1.0, 2.0, 5.0], help="Five fitting start times in seconds")
    ap.add_argument("--mapping-csv", default=None, help="Optional manual mapping CSV with columns group,file")
    ap.add_argument("--dry-run", action="store_true", help="Only show selected groups and matched files; do not fit")
    args = ap.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    book_path = Path(args.book).expanduser()
    if not book_path.is_absolute():
        book_path = root / book_path
    ca_dir = Path(args.ca_dir).expanduser()
    if not ca_dir.is_absolute():
        ca_dir = root / ca_dir
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = root / out_path
    mapping_path = Path(args.mapping_csv).expanduser() if args.mapping_csv else None

    if len(args.starts) != 5:
        raise ValueError("Please provide exactly five start times, e.g. --starts 0.2 0.5 1 2 5")

    meta = load_book(book_path)
    groups = sorted(meta["group"].unique())
    group_to_file, unmapped_files = discover_ca_files(ca_dir, groups, mapping_path)

    matched_groups = sorted(set(groups) & set(group_to_file))
    missing_groups = sorted(set(groups) - set(group_to_file))

    print(f"Selected groups from Book1: {len(groups)}")
    print(f"Matched CA files: {len(matched_groups)}")
    print(f"Missing groups: {missing_groups}")
    if unmapped_files:
        print(f"Unmapped/duplicate CA-like files: {len(unmapped_files)}")
        for p in unmapped_files[:20]:
            print(f"  unmapped_or_duplicate: {p}")
        if len(unmapped_files) > 20:
            print("  ...")

    if args.dry_run:
        preview = pd.DataFrame({
            "group": matched_groups,
            "file": [str(group_to_file[g]) for g in matched_groups],
        })
        print(preview.to_string(index=False))
        return

    rows = []
    meta_by_group = meta.set_index("group")
    for g in matched_groups:
        m = meta_by_group.loc[g].to_dict()
        file_path = group_to_file[g]
        try:
            trace = load_ca_trace(file_path)
            for start_s in args.starts:
                for model_name in ["beta", "beta_linear_drift"]:
                    res = fit_one(trace, start_s=start_s, model_name=model_name)
                    rows.append({
                        "group": g,
                        "substance": m["substance"],
                        "ca_potential": m["ca_potential"],
                        "polishing": m["polishing"],
                        "disconnected": m["disconnected"],
                        "day": m["day"],
                        "ca_file": str(file_path),
                        **res,
                    })
        except Exception as e:
            rows.append({
                "group": g,
                "substance": m.get("substance"),
                "ca_potential": m.get("ca_potential"),
                "polishing": m.get("polishing"),
                "disconnected": m.get("disconnected"),
                "day": m.get("day"),
                "ca_file": str(file_path),
                "success": False,
                "message": f"failed_to_load_or_fit: {repr(e)}",
            })

    result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame(columns=[
            "group", "substance", "ca_potential", "polishing", "disconnected", "day",
            "ca_file", "fit_start_s", "fit_end_s", "model", "A", "beta", "beta_se",
            "C", "M_linear_drift", "rmse", "r2", "n_points", "success", "message"
        ])
        print("No fit rows were produced. This usually means no CA files matched or all matched files failed to read.")
    else:
        result = result.sort_values(["group", "fit_start_s", "model"], na_position="last")
    result.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Also save missing group report for fixing filename mapping.
    missing_path = out_path.with_name(out_path.stem + "__missing_groups.csv")
    pd.DataFrame({"missing_group": missing_groups}).to_csv(missing_path, index=False)
    print(f"Saved missing-group report: {missing_path}")


if __name__ == "__main__":
    main()
