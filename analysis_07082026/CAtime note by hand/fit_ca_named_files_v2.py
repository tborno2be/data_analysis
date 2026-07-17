#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_ca_named_files_v2.py

分析一个文件夹里的 CA csv 文件。
每个 csv 文件作为一组数据，group 名直接使用文件名（不含 .csv）。

兼容这种 CA 文件格式：

    #,potential_V=0.4
    time,current
    5.063308e-02,9.764118e-06
    ...

对每个文件、每个起始时间，拟合两种模型：

1) normal beta:
   i(t) = A * x^(-beta) + C

2) beta + linear drift:
   i(t) = A * x^(-beta) + C + M*x

这里 x = time - fit_start_time + 一个很小的 eps，避免 x=0 发散。

用法：
    python fit_ca_named_files_v2.py

指定起始时间：
    python fit_ca_named_files_v2.py --starts 0.2 0.5 1 2 5 10

指定文件夹：
    python fit_ca_named_files_v2.py --data-dir ./analysis

只拟合到某个结束时间：
    python fit_ca_named_files_v2.py --end 30
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def normalize_colname(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "")


def looks_like_real_header(line: str) -> bool:
    """
    判断某一行是否像真正的数据表头。
    主要识别 time,current / time_s,current_a 之类。
    """
    low = line.strip().lower().replace(" ", "").replace("_", "")
    return ("time" in low or low.startswith("t,")) and (
        "current" in low or ",i" in low or "电流" in low
    )


def find_header_row(path: Path) -> int:
    """
    找真正的 header 行。
    你的文件第一行是 #,potential_V=0.4，所以应跳过第一行。
    """
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for idx, line in enumerate(f):
            if looks_like_real_header(line):
                return idx
    # 找不到就默认第一行，让后面的报错显示列名
    return 0


def detect_column(df: pd.DataFrame, kind: str) -> str:
    cols = list(df.columns)
    norm_map = {col: normalize_colname(col) for col in cols}

    if kind == "time":
        patterns = [
            r"^time",
            r"time",
            r"^t$",
            r"^ts$",
            r"seconds?",
            r"sec",
            r"时间",
        ]
    elif kind == "current":
        patterns = [
            r"current",
            r"^i$",
            r"^ia$",
            r"amp",
            r"电流",
        ]
    else:
        raise ValueError(f"Unknown kind: {kind}")

    for pat in patterns:
        for col, ncol in norm_map.items():
            if re.search(pat, ncol, flags=re.IGNORECASE):
                return col

    raise ValueError(
        f"Cannot auto-detect {kind} column. Columns are: {cols}. "
        f"Please use --{kind}-col."
    )


def read_ca_csv(path: Path, time_col: Optional[str], current_col: Optional[str]) -> pd.DataFrame:
    header_row = find_header_row(path)

    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        header=0,
        skiprows=header_row,
        encoding="utf-8-sig",
    )

    # 删掉全空列/全空行
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

    if df.empty:
        raise ValueError("CSV is empty after reading.")

    tcol = time_col or detect_column(df, "time")
    icol = current_col or detect_column(df, "current")

    out = pd.DataFrame({
        "time_s": pd.to_numeric(df[tcol], errors="coerce"),
        "current_a": pd.to_numeric(df[icol], errors="coerce"),
    })

    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.sort_values("time_s").drop_duplicates(subset=["time_s"])

    if out.empty:
        raise ValueError("No numeric time/current data after cleaning.")

    return out


def beta_model(x: np.ndarray, A: float, beta: float, C: float) -> np.ndarray:
    return A * np.power(x, -beta) + C


def beta_linear_model(x: np.ndarray, A: float, beta: float, C: float, M: float) -> np.ndarray:
    return A * np.power(x, -beta) + C + M * x


def guess_initial_params(x: np.ndarray, y: np.ndarray, with_linear: bool):
    tail_n = max(3, len(y) // 10)
    c0 = float(np.nanmedian(y[-tail_n:]))

    amp = float(y[0] - c0)
    if not np.isfinite(amp) or abs(amp) < 1e-15:
        amp = float(np.nanmax(y) - np.nanmin(y))
    if not np.isfinite(amp) or abs(amp) < 1e-15:
        amp = 1e-9

    beta0 = 0.5

    if with_linear:
        try:
            m0 = float(np.polyfit(x, y, 1)[0])
        except Exception:
            m0 = 0.0
        p0 = [amp, beta0, c0, m0]
        bounds = ([-np.inf, 0.0, -np.inf, -np.inf], [np.inf, 2.0, np.inf, np.inf])
    else:
        p0 = [amp, beta0, c0]
        bounds = ([-np.inf, 0.0, -np.inf], [np.inf, 2.0, np.inf])

    return p0, bounds


def fit_one_window(
    time_s: np.ndarray,
    current_a: np.ndarray,
    start_s: float,
    end_s: Optional[float],
    with_linear: bool,
    min_points: int,
) -> dict:
    if end_s is None:
        mask = time_s >= start_s
        actual_end_s = float(np.nanmax(time_s))
    else:
        mask = (time_s >= start_s) & (time_s <= end_s)
        actual_end_s = float(end_s)

    t = time_s[mask]
    y = current_a[mask]

    model_name = "beta_plus_linear_drift" if with_linear else "beta"

    if len(t) < min_points:
        return {
            "model": model_name,
            "fit_start_s": float(start_s),
            "fit_end_s": actual_end_s,
            "n_points": int(len(t)),
            "success": False,
            "message": f"not enough points: {len(t)} < {min_points}",
        }

    # 用相对时间，避免 t 很大导致拟合相关性更强；同时避开 x=0 发散。
    x = t - float(t[0])
    steps = np.diff(x)
    steps = steps[steps > 0]
    eps = float(np.nanmedian(steps)) if len(steps) else 1e-6
    if not np.isfinite(eps) or eps <= 0:
        eps = 1e-6
    x = x + eps

    model = beta_linear_model if with_linear else beta_model

    try:
        p0, bounds = guess_initial_params(x, y, with_linear)
        popt, pcov = curve_fit(
            model,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=100000,
        )

        yhat = model(x, *popt)
        resid = y - yhat
        rmse = float(np.sqrt(np.mean(resid ** 2)))

        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        try:
            perr = np.sqrt(np.diag(pcov))
        except Exception:
            perr = [np.nan] * len(popt)

        if with_linear:
            A, beta, C, M = popt
            A_se, beta_se, C_se, M_se = perr
        else:
            A, beta, C = popt
            A_se, beta_se, C_se = perr
            M = np.nan
            M_se = np.nan

        return {
            "model": model_name,
            "fit_start_s": float(start_s),
            "fit_end_s": actual_end_s,
            "n_points": int(len(t)),
            "A": float(A),
            "A_se": float(A_se),
            "beta": float(beta),
            "beta_se": float(beta_se),
            "C": float(C),
            "C_se": float(C_se),
            "M_linear_drift": float(M),
            "M_linear_drift_se": float(M_se),
            "rmse": rmse,
            "r2": r2,
            "success": True,
            "message": "",
        }

    except Exception as e:
        return {
            "model": model_name,
            "fit_start_s": float(start_s),
            "fit_end_s": actual_end_s,
            "n_points": int(len(t)),
            "success": False,
            "message": repr(e),
        }


def natural_sort_key(path: Path):
    s = path.stem
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p for p in parts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="放 CA csv 的文件夹。默认当前文件夹。")
    parser.add_argument("--out", default="ca_beta_named_results.csv", help="输出 csv 文件名。")
    parser.add_argument("--starts", nargs="+", type=float, default=[0.2, 0.5, 1, 2, 5],
                        help="拟合起始时间，单位秒。默认 0.2 0.5 1 2 5")
    parser.add_argument("--end", type=float, default=None, help="拟合结束时间，单位秒。默认到最后。")
    parser.add_argument("--time-col", default=None, help="手动指定时间列名。")
    parser.add_argument("--current-col", default=None, help="手动指定电流列名。")
    parser.add_argument("--recursive", action="store_true", help="递归读取子文件夹。")
    parser.add_argument("--pattern", default="*.csv", help="文件匹配模式，默认 *.csv。")
    parser.add_argument("--min-points", type=int, default=20, help="每个窗口最少点数，默认 20。")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    files = list(data_dir.rglob(args.pattern) if args.recursive else data_dir.glob(args.pattern))
    files = sorted(files, key=natural_sort_key)

    if not files:
        raise FileNotFoundError(f"No csv files found in {data_dir} with pattern {args.pattern!r}")

    rows = []

    for file in files:
        group_name = file.stem

        # 避免把上一次输出结果又读进去
        if file.name == Path(args.out).name:
            continue

        try:
            df = read_ca_csv(file, args.time_col, args.current_col)
            time_s = df["time_s"].to_numpy(dtype=float)
            current_a = df["current_a"].to_numpy(dtype=float)

            for start_s in args.starts:
                for with_linear in [False, True]:
                    fit = fit_one_window(
                        time_s=time_s,
                        current_a=current_a,
                        start_s=start_s,
                        end_s=args.end,
                        with_linear=with_linear,
                        min_points=args.min_points,
                    )
                    rows.append({
                        "group": group_name,
                        "file": str(file),
                        "time_min_s": float(np.nanmin(time_s)),
                        "time_max_s": float(np.nanmax(time_s)),
                        "current_min_a": float(np.nanmin(current_a)),
                        "current_max_a": float(np.nanmax(current_a)),
                        **fit,
                    })

        except Exception as e:
            rows.append({
                "group": group_name,
                "file": str(file),
                "success": False,
                "message": repr(e),
            })

    result = pd.DataFrame(rows)

    preferred_cols = [
        "group", "file", "model", "fit_start_s", "fit_end_s", "n_points",
        "A", "A_se", "beta", "beta_se", "C", "C_se",
        "M_linear_drift", "M_linear_drift_se",
        "rmse", "r2", "success", "message",
        "time_min_s", "time_max_s", "current_min_a", "current_max_a",
    ]
    existing = [c for c in preferred_cols if c in result.columns]
    rest = [c for c in result.columns if c not in existing]
    result = result[existing + rest]

    if {"group", "fit_start_s", "model"}.issubset(result.columns):
        result = result.sort_values(["group", "fit_start_s", "model"], kind="stable")

    out_path = Path(args.out).expanduser().resolve()
    result.to_csv(out_path, index=False)

    print(f"Input folder: {data_dir}")
    print(f"CSV files found: {len(files)}")
    print(f"Rows written: {len(result)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
