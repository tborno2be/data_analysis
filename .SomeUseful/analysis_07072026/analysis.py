"""Analysis functions for the test dataset: loading, CA alpha fit, fractional transform, background subtraction, fault features."""

from __future__ import annotations

from pathlib import Path
from korobka.cyclic_voltametry.EmStat4X import reconstruct_time

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma as sp_gamma
import csv
import logging
LOG = logging.getLogger(__name__)

# -------------------------------loaders (test-data format only)------------------------------

def load_cv(path: str | Path) -> dict:
    """Read a CV data.csv into {'equil': (E, i), 'scans': {k: (E, i, t_or_None)}}."""
    equil_E, equil_i = [], []
    scans: dict[int, list] = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            scan, e, i, t = line.rstrip("\n").split(",")
            if scan == "":
                equil_E.append(float(e))
                equil_i.append(float(i))
                continue
            scans.setdefault(int(scan), []).append((float(e), float(i), float(t) if t else None))
    out_scans = {}
    for k, rows in scans.items():
        E = np.array([r[0] for r in rows])
        i = np.array([r[1] for r in rows])
        t = None if rows[0][2] is None else np.array([r[2] for r in rows])
        out_scans[k] = (E, i, t)
    return {"equil": (np.array(equil_E), np.array(equil_i)), "scans": out_scans}


def load_ca(path: str | Path) -> tuple[float, np.ndarray, np.ndarray]:
    """Read a CA data.csv into (potential_v, t, i)."""
    with open(path, encoding="utf-8") as f:
        potential_v = float(f.readline().split("=")[1])
        next(f)
        rows = [line.rstrip("\n").split(",") for line in f]
    t = np.array([float(r[0]) for r in rows])
    i = np.array([float(r[1]) for r in rows])
    return potential_v, t, i


# -------------------------------CA analysis (table one)------------------------------


CANDIDATE_STARTS = [1, 3, 5, 7, 9, 12, 15, 19]
PLATEAU_TOL = 0.01
FALLBACK_T_MIN = 0.4


def group_metadata() -> dict[int, dict]:
    """Group number -> substance/potential/polish/disconnect/day, per the 115-group test plan."""
    meta = {}
    pots = [0.0, 0.1, 0.2, 0.3, 0.4]
    for n in range(1, 116):
        if n <= 5:
            sub = "tapwater"
        elif n <= 10 or 21 <= n <= 50 or 76 <= n <= 95:
            sub = "blank"
        else:
            sub = "sample"
        if 36 <= n <= 50 or 61 <= n <= 75:
            disc = ["working", "counter", "reference"][((n - 36) if n <= 50 else (n - 61)) // 5]
        else:
            disc = "none"
        meta[n] = {
            "group": n,
            "substance": sub,
            "potential_nominal": pots[(n - 1) % 5] if n <= 75 else 0.4,
            "polished": n >= 26,
            "disconnect": disc,
            "day": "Y" if n <= 75 else "T",
        }
    return meta


def ca_raw_stats(t: np.ndarray, i: np.ndarray) -> dict:
    """Model-free descriptors, valid even when fitting fails."""
    counts = np.unique(np.round(i, 12), return_counts=True)[1]
    if np.all(i > 0):
        sign = "pos"
    elif np.all(i < 0):
        sign = "neg"
    else:
        sign = "mixed"
    return {
        "n_points": len(i),
        "duration_s": float(t[-1] - t[0]),
        "first_i": float(i[0]),
        "tail_mean": float(np.mean(i[-10:])),
        "i_sign": sign,
        "noise_mad": succ_diff_mad(i),
        "saturation_frac": float(counts.max() / len(i)),
    }


def detect_jump(i: np.ndarray, window: int = 20, rel: float = 0.3) -> int:
    """Index right after the last >30% relative jump in the first `window` points (0 if none)."""
    end = min(window, len(i) - 1)
    jump_after = 0
    for k in range(end):
        if abs(i[k]) > 0 and abs(i[k + 1] - i[k]) > rel * abs(i[k]):
            jump_after = k + 1
    return jump_after


def _fit_power(t: np.ndarray, i: np.ndarray) -> dict | None:
    """Single-window fit of i = a*t^(-beta) + c; None when it does not converge."""
    def model(t, a, beta, c):
        return a * t ** (-beta) + c

    c0 = float(np.mean(i[-10:]))
    a0 = float((i[0] - c0) * t[0] ** 0.5)
    try:
        p, _ = curve_fit(model, t, i, p0=[a0, 0.5, c0],
                         bounds=([-np.inf, 0.05, -np.inf], [np.inf, 1.5, np.inf]), maxfev=10000)
    except (RuntimeError, ValueError):
        return None
    a, beta, c = (float(v) for v in p)
    resid = i - model(t, a, beta, c)
    ss_tot = float(np.sum((i - np.mean(i)) ** 2))
    return {"a": a, "beta": beta, "c": c,
            "r2": 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan,
            "resid_mad": float(np.median(np.abs(resid - np.median(resid)))) * 1.4826}


def fit_ca_beta(t: np.ndarray, i: np.ndarray) -> dict:
    """Charge-current skip via beta-plateau start scan, jump sentinel, fixed-t_min fallback."""
    jump_after = detect_jump(i)
    starts = [s for s in CANDIDATE_STARTS if s >= jump_after and s < len(i) - 15]
    fits = [(s, _fit_power(t[s:], i[s:])) for s in starts]
    betas = [(s, f) for s, f in fits if f is not None]

    chosen = None
    for j in range(len(betas) - 2):
        d1 = abs(betas[j + 1][1]["beta"] - betas[j][1]["beta"])
        d2 = abs(betas[j + 2][1]["beta"] - betas[j + 1][1]["beta"])
        if d1 < PLATEAU_TOL and d2 < PLATEAU_TOL:
            chosen = betas[j]
            break

    if chosen is not None:
        s, f = chosen
        return {**f, "t_min_used": float(t[s]), "n_skipped": s,
                "plateau_found": True, "jump_index": jump_after, "notes": ""}

    m = t >= FALLBACK_T_MIN
    f = _fit_power(t[m], i[m]) if m.sum() > 15 else None
    base = {"t_min_used": FALLBACK_T_MIN, "n_skipped": int((~m).sum()) if m.sum() > 15 else len(i),
            "plateau_found": False, "jump_index": jump_after}
    if f is None:
        return {**base, "a": np.nan, "beta": np.nan, "c": np.nan, "r2": np.nan,
                "resid_mad": np.nan, "notes": "no plateau; fallback fit failed"}
    return {**base, **f, "notes": "no plateau; fallback t_min"}


def build_ca_table(ca_root: str | Path, out_csv: str | Path) -> list[dict]:
    """Walk CA/<n>/data.csv for all 115 groups and write table one."""
    ca_root = Path(ca_root)
    rows = []
    for n, m in group_metadata().items():
        path = ca_root / str(n) / "data.csv"
        row = dict(m)
        if not path.exists():
            row["notes"] = "file missing"
            rows.append(row)
            continue
        potential_v, t, i = load_ca(path)
        row["potential_file"] = potential_v
        row.update(ca_raw_stats(t, i))
        row.update(fit_ca_beta(t, i))
        row.update({f"{k}_lockc": v for k, v in fit_ca_beta_lockc(t, i).items()})
        rows.append(row)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    LOG.info("CA table written to %s (%d rows)", out_csv, len(rows))
    return rows

# -------------------------------fractional transform------------------------------

def fracdiff(i: np.ndarray, dt: float, alpha: float) -> np.ndarray:
    """Riemann-Liouville (Oldham-Spanier) differintegral of order alpha (negative alpha integrates)."""
    n = len(i)
    j = np.arange(n, dtype=float)
    w = np.empty(n)
    w[0] = 1.0 / sp_gamma(2.0 - alpha)
    w[1:] = ((j[1:] + 1.0) ** (1.0 - alpha) - 2.0 * j[1:] ** (1.0 - alpha)
             + (j[1:] - 1.0) ** (1.0 - alpha)) / sp_gamma(2.0 - alpha)
    out = dt ** (-alpha) * np.convolve(i, w)[:n]
    out[0] = 0.0
    return out

# -------------------------------background subtraction------------------------------

def subtract_scans(sample: dict, blank: dict) -> tuple[dict, dict]:
    """Per-scan sample-minus-blank; raises on scan/point-count mismatch. Returns (corrected, diagnostics)."""
    if sample["scans"].keys() != blank["scans"].keys():
        raise ValueError("Scan sets differ between sample and blank.")
    corrected, diag = {}, {}
    for k in sample["scans"]:
        E_s, i_s, t_s = sample["scans"][k]
        E_b, i_b, _ = blank["scans"][k]
        if len(i_s) != len(i_b):
            raise ValueError(f"Scan {k}: point counts differ ({len(i_s)} vs {len(i_b)}).")
        resid = i_s - i_b
        corrected[k] = (E_s, resid, t_s)
        diag[k] = {"rms": float(np.sqrt(np.mean(resid ** 2))),
                   "max_abs": float(np.max(np.abs(resid))),
                   "sd_mad": succ_diff_mad(resid)}
    return corrected, diag

# -------------------------------fault features------------------------------

def succ_diff_mad(i: np.ndarray) -> float:
    """Noise sigma from successive differences via MAD."""
    d = np.diff(i)
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2))


def feature_vector(cv: dict) -> dict:
    """Connection/quality features for one CV file."""
    eq_E, eq_i = cv["equil"]
    d_eq = np.diff(eq_i)
    all_i = np.concatenate([cv["scans"][k][1] for k in sorted(cv["scans"])])
    E0, i0, _ = cv["scans"][0]
    counts = np.unique(np.round(all_i, 12), return_counts=True)[1]

    feats = {
        "equil_monotonic": float(np.mean(np.sign(d_eq) == np.sign(np.median(d_eq)))) if len(d_eq) else np.nan,
        "equil_decay_ratio": float(eq_i[0] / eq_i[-1]) if len(eq_i) and eq_i[-1] != 0 else np.nan,
        "equil_range": float(np.ptp(eq_i)) if len(eq_i) else np.nan,
        "max_abs_i": float(np.max(np.abs(all_i))),
        "rms_i": float(np.sqrt(np.mean(all_i ** 2))),
        "saturation_frac": float(counts.max() / len(all_i)),
        "envelope": float(np.mean(np.abs(i0[:len(i0) // 2] - i0[:len(i0) // 2][::-1]))),
        "noise_mad": succ_diff_mad(all_i),
        "E_span_ok": bool(np.ptp(E0) > 0.9 * 0.6),
    }
    if 1 in cv["scans"]:
        i1 = cv["scans"][1][1]
        n = min(len(i0), len(i1))
        feats["scan_corr"] = float(np.corrcoef(i0[:n], i1[:n])[0, 1])
    else:
        feats["scan_corr"] = np.nan
    return feats




def fit_ca_beta_lockc(t: np.ndarray, i: np.ndarray, tail_n: int = 20) -> dict:
    """Two-parameter fit i = a*t^(-beta) + c with c fixed to the tail mean; same start-scan skip."""
    c_fixed = float(np.mean(i[-tail_n:]))

    def model(t, a, beta):
        return a * t ** (-beta) + c_fixed

    def fit_window(tw, iw):
        a0 = float((iw[0] - c_fixed) * tw[0] ** 0.5)
        try:
            p, _ = curve_fit(model, tw, iw, p0=[a0, 0.5],
                             bounds=([-np.inf, 0.05], [np.inf, 1.5]), maxfev=10000)
        except (RuntimeError, ValueError):
            return None
        a, beta = float(p[0]), float(p[1])
        resid = iw - model(tw, a, beta)
        ss_tot = float(np.sum((iw - np.mean(iw)) ** 2))
        return {"a": a, "beta": beta, "c": c_fixed,
                "r2": 1 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan,
                "resid_mad": float(np.median(np.abs(resid - np.median(resid)))) * 1.4826}

    jump_after = detect_jump(i)
    starts = [s for s in CANDIDATE_STARTS if s >= jump_after and s < len(i) - 15]
    fits = [(s, fit_window(t[s:], i[s:])) for s in starts]
    betas = [(s, f) for s, f in fits if f is not None]

    chosen = None
    for j in range(len(betas) - 2):
        if (abs(betas[j + 1][1]["beta"] - betas[j][1]["beta"]) < PLATEAU_TOL
                and abs(betas[j + 2][1]["beta"] - betas[j + 1][1]["beta"]) < PLATEAU_TOL):
            chosen = betas[j]
            break

    if chosen is not None:
        s, f = chosen
        return {**f, "t_min_used": float(t[s]), "n_skipped": s, "plateau_found": True,
                "jump_index": jump_after, "notes": "locked-c"}

    m = t >= FALLBACK_T_MIN
    f = fit_window(t[m], i[m]) if m.sum() > 15 else None
    base = {"t_min_used": FALLBACK_T_MIN, "n_skipped": int((~m).sum()) if m.sum() > 15 else len(i),
            "plateau_found": False, "jump_index": jump_after}
    if f is None:
        return {**base, "a": np.nan, "beta": np.nan, "c": c_fixed, "r2": np.nan,
                "resid_mad": np.nan, "notes": "locked-c; fallback failed"}
    return {**base, **f, "notes": "locked-c; fallback t_min"}

# -------------------------------CV analysis (tables two & three)------------------------------

def cv_split_sweep(E, i):
    """Rising branch (neg vertex -> pos vertex, anodic) and falling branch (pos vertex -> end, cathodic)."""
    v_neg = int(np.argmin(E))
    v_pos = int(np.argmax(E))
    return E[v_neg:v_pos + 1], i[v_neg:v_pos + 1], E[v_pos:], i[v_pos:]


def _peak_on_branch(E, i, anodic, noise_k=5.0, window=(0.05, 0.25)):
    """Peak with linear baseline, gated by noise threshold and expected potential window."""
    if len(E) < 5:
        return {"Ep": np.nan, "ip": np.nan}
    baseline = np.interp(E, [E[0], E[-1]], [i[0], i[-1]])
    corr = i - baseline
    idx = int(np.argmax(corr)) if anodic else int(np.argmin(corr))
    if idx in (0, len(E) - 1):
        return {"Ep": np.nan, "ip": np.nan}
    noise = succ_diff_mad(corr)
    if abs(corr[idx]) < noise_k * noise or not (window[0] <= E[idx] <= window[1]):
        return {"Ep": np.nan, "ip": np.nan}
    return {"Ep": float(E[idx]), "ip": float(corr[idx])}


def cv_scan_peaks(E: np.ndarray, i: np.ndarray) -> dict:
    """Anodic peak on forward branch, cathodic on reverse; E_half, dEp, ratio. All NaN-safe."""
    Ef, If, Er, Ir = cv_split_sweep(E, i)
    a = _peak_on_branch(Ef, If, anodic=True)
    c = _peak_on_branch(Er, Ir, anodic=False)
    E_half = (a["Ep"] + c["Ep"]) / 2 if not (np.isnan(a["Ep"]) or np.isnan(c["Ep"])) else np.nan
    dEp = a["Ep"] - c["Ep"] if not (np.isnan(a["Ep"]) or np.isnan(c["Ep"])) else np.nan
    ratio = abs(c["ip"] / a["ip"]) if not (np.isnan(a["ip"]) or np.isnan(c["ip"]) or a["ip"] == 0) else np.nan
    return {"Epa": a["Ep"], "Epc": c["Ep"], "ipa": a["ip"], "ipc": c["ip"],
            "E_half": E_half, "dEp": dEp, "ip_ratio": ratio}


def cv_scan_background(E: np.ndarray, i: np.ndarray, win: float = 0.05) -> dict:
    """Capacitive envelope near E=0, current at both vertices, over a non-Faradaic window."""
    Ef, If, Er, Ir = cv_split_sweep(E, i)
    mask_f = np.abs(Ef) < win
    mask_r = np.abs(Er) < win
    if mask_f.any() and mask_r.any():
        if_near = float(np.mean(If[mask_f]))
        ir_near = float(np.mean(Ir[mask_r]))
        envelope = abs(if_near - ir_near)
    else:
        envelope = np.nan
    return {"envelope": envelope,
            "i_vertex_neg": float(i[int(np.argmin(E))]),
            "i_vertex_pos": float(i[int(np.argmax(E))])}


def cv_interscan_drift(scans: dict) -> dict:
    """RMS of (scan k - scan 0) over the common length, for k = 1, 2."""
    if 0 not in scans:
        return {"drift_1": np.nan, "drift_2": np.nan}
    i0 = scans[0][1]
    out = {}
    for k in (1, 2):
        if k in scans:
            ik = scans[k][1]
            n = min(len(i0), len(ik))
            out[f"drift_{k}"] = float(np.sqrt(np.mean((ik[:n] - i0[:n]) ** 2)))
        else:
            out[f"drift_{k}"] = np.nan
    return out


def build_cv_tables(cv_root: str | Path, out_peaks: str | Path, out_feats: str | Path) -> tuple[list, list]:
    """Walk CV/<n>/data.csv for all groups; write table two (per-scan peaks+background) and table three (features)."""
    cv_root = Path(cv_root)
    Path(out_peaks).parent.mkdir(parents=True, exist_ok=True)
    peak_rows, feat_rows = [], []

    for n, m in group_metadata().items():
        path = cv_root / str(n) / "data.csv"
        if not path.exists():
            peak_rows.append({**m, "notes": "file missing"})
            feat_rows.append({**m, "notes": "file missing"})
            continue
        cv = load_cv(path)

        prow = dict(m)
        prow["n_scans_parsed"] = len(cv["scans"])
        for k in sorted(cv["scans"]):
            E, i, _ = cv["scans"][k]
            for key, val in cv_scan_peaks(E, i).items():
                prow[f"{key}_s{k}"] = val
            for key, val in cv_scan_background(E, i).items():
                prow[f"{key}_s{k}"] = val
        prow.update(cv_interscan_drift(cv["scans"]))
        peak_rows.append(prow)

        frow = dict(m)
        frow.update(feature_vector(cv))
        feat_rows.append(frow)

    for rows, out in ((peak_rows, out_peaks), (feat_rows, out_feats)):
        fields = sorted({k for r in rows for k in r})
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        LOG.info("Wrote %s (%d rows)", out, len(rows))
    return peak_rows, feat_rows

# -------------------------------CV semiderivative diagnostics (RL, time-axis)------------------------------

from scipy.signal import find_peaks
from lmfit.models import Pearson4Model, LinearModel

SD_SCALE = 1e6
FIT_HALFWIN_S = 1.5
FIND_DISTANCE = 15
FC_E0 = 0.19  # FcMeOH 1 mM in KCl vs this cell's reference (accuracy target for E_half)


def semideriv_full(i: np.ndarray, dt: float, order: float = 0.5) -> np.ndarray:
    """Time-domain RL semiderivative of the FULL scan current (scaled), whole sweep at once - never a single branch."""
    return fracdiff(i * SD_SCALE, dt, order)


def _branch_masks(E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward (anodic, dE/dt>0) and reverse (cathodic, dE/dt<0) by scan direction in time, not by E ordering."""
    d = np.sign(np.diff(E))
    d = np.append(d, d[-1] if len(d) else 0.0)
    return d > 0, d < 0


# ---- three peak-current readings, compared against the known Fc E0 ----

def simple_ip_linear(E: np.ndarray, i: np.ndarray, anodic: bool) -> dict:
    """Reading 1: two-vertex branch split, monotonic branch, two-endpoint linear baseline, NO gates.

    The un-gated traditional target - always returns an extremum (even a background hump),
    which is precisely what makes it the 'naive traditional method' benchmark.
    """
    Ef, If, Er, Ir = cv_split_sweep(E, i)
    Eb, ib = (Ef, If) if anodic else (Er, Ir)
    if len(Eb) < 5:
        return {"Ep": np.nan, "ip": np.nan}
    baseline = np.interp(Eb, [Eb[0], Eb[-1]], [ib[0], ib[-1]])
    corr = ib - baseline
    idx = int(np.argmax(corr)) if anodic else int(np.argmin(corr))
    return {"Ep": float(Eb[idx]), "ip": float(corr[idx])}


def simple_ip_findpeaks(E: np.ndarray, i: np.ndarray, anodic: bool, prom_frac: float = 0.5) -> dict:
    """Reading 2: same branch + linear baseline, but locate the peak with find_peaks (prominence gate).

    Differs from reading 1 only by the prominence gate: returns NaN when no branch peak is
    prominent enough, so it is stricter than the naive extremum.
    """
    Ef, If, Er, Ir = cv_split_sweep(E, i)
    Eb, ib = (Ef, If) if anodic else (Er, Ir)
    if len(Eb) < 5:
        return {"Ep": np.nan, "ip": np.nan}
    baseline = np.interp(Eb, [Eb[0], Eb[-1]], [ib[0], ib[-1]])
    corr = ib - baseline
    y = corr if anodic else -corr
    pk, _ = find_peaks(y, prominence=np.std(y) * prom_frac)
    if len(pk) == 0:
        return {"Ep": np.nan, "ip": np.nan}
    main = int(pk[np.argmax(y[pk])])
    return {"Ep": float(Eb[main]), "ip": float(corr[main])}


def _fit_pearson_peak(t: np.ndarray, sd: np.ndarray, E: np.ndarray, dt: float,
                      anodic: bool, noise_k: float = 5.0, order: float = 0.5) -> dict | None:
    """Reading 3 core: locate on time axis (find_peaks), fit LINEAR BACKGROUND FIRST then Pearson IV;
    ip via -order RL semi-integral LSV step (max-min). Background term is retained even after blank subtraction
    to absorb the un-subtractable O2 residual - never a bare Pearson fit."""
    y = sd if anodic else -sd
    if y.size < 10 or y.max() <= 0:
        return None
    noise = succ_diff_mad(y)
    pk, _ = find_peaks(y, height=max(0.4 * y.max(), noise_k * noise), distance=FIND_DISTANCE)
    if len(pk) == 0:
        return None
    main = int(pk[np.argmax(y[pk])])
    tpk = t[main]
    win = np.abs(t - tpk) < FIT_HALFWIN_S
    xw, yw = t[win], y[win]
    if len(xw) < 10:
        return None

    # 1) fit linear background on the window edges, freeze it
    bg = LinearModel(prefix="bg_")
    edge_x = np.r_[xw[:3], xw[-3:]]
    edge_y = np.r_[yw[:3], yw[-3:]]
    bg_pars = bg.guess(edge_y, x=edge_x)
    bg_line = bg.eval(bg_pars, x=xw)
    yw_c = yw - bg_line

    # 2) fit Pearson IV on the background-freed peak
    pk_m = Pearson4Model(prefix="pk_")
    pk_pars = pk_m.guess(yw_c, x=xw)
    pk_pars["pk_center"].set(value=tpk, min=tpk - 0.5, max=tpk + 0.5)
    pk_pars["pk_expon"].set(value=1.5, min=0.6)
    pk_pars["pk_skew"].set(value=0.0)
    try:
        res = pk_m.fit(yw_c, pk_pars, x=xw)
    except Exception:
        return None
    fwhm = 2.0 * res.params["pk_sigma"].value * np.sqrt(
        2 ** (1.0 / res.params["pk_expon"].value) - 1.0)
    if fwhm > 2.0 * FIT_HALFWIN_S:
        return None

    Ep = float(np.interp(res.params["pk_center"].value, t, E))
    peak_full = res.eval(x=t)
    lsv = fracdiff(peak_full, dt, -order) / SD_SCALE
    ip_step = float(lsv.max() - lsv.min())
    sign = 1.0 if anodic else -1.0
    ss_res = float(np.sum(res.residual ** 2))
    ss_tot = float(np.sum((yw_c - np.mean(yw_c)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"Ep": Ep, "ip": sign * ip_step,
            "sd_height": float(res.params["pk_height"].value),
            "sigma": float(res.params["pk_sigma"].value),
            "expon": float(res.params["pk_expon"].value),
            "fwhm": float(fwhm),
            "skew": float(res.params["pk_skew"].value),
            "redchi": float(res.redchi),
            "r2": float(r2),
            "peak_over_noise": float(y[main] / noise) if noise else np.nan}


def mean_blank_scans(blank_ids: list[int], cv_root: str | Path) -> dict:
    """Point-wise mean blank per scan key over several blank groups; raises on point-count mismatch (no silent interp)."""
    cv_root = Path(cv_root)
    cvs = [load_cv(cv_root / str(n) / "data.csv") for n in blank_ids]
    keys = set(cvs[0]["scans"])
    for c in cvs:
        if set(c["scans"]) != keys:
            raise ValueError("Blank groups differ in scan set.")
    out = {}
    for k in sorted(keys):
        arrs = [c["scans"][k][1] for c in cvs]
        n0 = len(arrs[0])
        for a in arrs:
            if len(a) != n0:
                raise ValueError(f"Blank scan {k}: point counts differ across groups.")
        out[k] = (cvs[0]["scans"][k][0], np.mean(arrs, axis=0))
    return out


def sample_scan_three_readings(E_s: np.ndarray, i_s: np.ndarray, blank_scan: tuple, dt: float) -> dict:
    """One sample scan, three peak readings side by side, each compared to FC_E0.

    Reading 1 (linear) and 2 (find_peaks) act on the RAW CV. Reading 3 (pearson) acts on the
    mean-blank-subtracted time-axis RL semiderivative. E_half from anodic+cathodic midpoint.
    """
    E_b, i_b = blank_scan
    n = min(len(i_s), len(i_b))
    if n < 30:
        return {}
    E, i_s, i_b = E_s[:n], i_s[:n], i_b[:n]
    t = np.asarray(reconstruct_time(n, dt), dtype=float)

    out = {}

    # readings 1 & 2 on raw CV
    for name, fn in (("lin", simple_ip_linear), ("fp", simple_ip_findpeaks)):
        a = fn(E, i_s, anodic=True)
        c = fn(E, i_s, anodic=False)
        out[f"{name}_ipa"] = a["ip"]
        out[f"{name}_ipc"] = c["ip"]
        out[f"{name}_Epa"] = a["Ep"]
        out[f"{name}_Epc"] = c["Ep"]
        if not (np.isnan(a["Ep"]) or np.isnan(c["Ep"])):
            out[f"{name}_Ehalf"] = (a["Ep"] + c["Ep"]) / 2
            out[f"{name}_Ehalf_err"] = out[f"{name}_Ehalf"] - FC_E0
        else:
            out[f"{name}_Ehalf"] = np.nan
            out[f"{name}_Ehalf_err"] = np.nan

    # reading 3 on blank-subtracted semiderivative
    sd = semideriv_full(i_s, dt) - semideriv_full(i_b, dt)
    fwd, rev = _branch_masks(E)
    an = _fit_pearson_peak(t[fwd], sd[fwd], E[fwd], dt, anodic=True)
    ca = _fit_pearson_peak(t[rev], sd[rev], E[rev], dt, anodic=False)
    if an:
        out.update({f"pearson_a_{k}": v for k, v in an.items()})
    if ca:
        out.update({f"pearson_c_{k}": v for k, v in ca.items()})
    if an and ca:
        out["pearson_Ehalf"] = (an["Ep"] + ca["Ep"]) / 2
        out["pearson_Ehalf_err"] = out["pearson_Ehalf"] - FC_E0
        out["pearson_ip_ratio"] = abs(ca["ip"] / an["ip"]) if an["ip"] else np.nan
    else:
        out["pearson_Ehalf"] = np.nan
        out["pearson_Ehalf_err"] = np.nan
    return out


def blank_residual_peaks(E: np.ndarray, i: np.ndarray, base_scan: tuple,
                         dt: float, noise_k: float = 5.0, neg_win: tuple = (-0.30, -0.05)) -> dict:
    """Residual of one blank scan RELATIVE TO the batch baseline blank: LOCAL peaks + GLOBAL magnitude.

    Baseline = first blank of each polish state. Subtract in the CURRENT domain first (matches the
    baseline's physical measurement), THEN semiderivative. Common background (O2, double layer,
    sawtooth) cancels against baseline; baseline-minus-itself is the empty anchor.

    Two complementary readouts, because they catch different drifts:
      * LOCAL peaks (find_peaks on the residual semiderivative) - a NEW peak that grew relative to
        baseline (contamination / a new species). Blind to a smooth overall shift.
      * GLOBAL magnitude (residual RMS in the current domain, plus mean residual over the negative
        O2 window) - an OVERALL baseline rise/fall, e.g. dissolved O2 increasing across a day. A
        smooth ramp produces no local semiderivative peak, so peaks alone would falsely report
        'no change'; the global terms make that visible.

    Returns {"peaks": [...], "resid_rms": float, "resid_neg_mean": float, "resid_max_abs": float}.
    """
    E_b, i_b = base_scan
    n = min(len(i), len(i_b))
    E, i_t, i_b = E[:n], i[:n], i_b[:n]
    resid = i_t - i_b                       # current-domain residual against baseline (先减)

    # global magnitude (catches overall background shift the semiderivative peak-finder misses)
    neg = (E >= neg_win[0]) & (E <= neg_win[1])
    resid_rms = float(np.sqrt(np.mean(resid ** 2)))
    resid_neg_mean = float(np.mean(resid[neg])) if neg.any() else np.nan
    resid_max_abs = float(np.max(np.abs(resid)))

    # local peaks on the residual semiderivative (catches newly grown peaks)
    sd = semideriv_full(resid, dt)          # then transform (后变换)
    fwd, rev = _branch_masks(E)
    peaks = []
    for anodic, mask, label in ((True, fwd, "anodic"), (False, rev, "cathodic")):
        y = sd[mask] if anodic else -sd[mask]
        Em = E[mask]
        if y.size < 10:
            continue
        noise = succ_diff_mad(y)
        idx, _ = find_peaks(y, height=noise_k * noise, distance=FIND_DISTANCE)
        for p in idx:
            peaks.append({"branch": label,
                          "E_peak": float(Em[p]),
                          "sd_height": float(y[p]),
                          "peak_over_noise": float(y[p] / noise) if noise else np.nan})
    return {"peaks": peaks, "resid_rms": resid_rms,
            "resid_neg_mean": resid_neg_mean, "resid_max_abs": resid_max_abs}


def _baseline_ids(meta: dict) -> dict:
    """Baseline blank per polish state, anchored to YESTERDAY's first blank and shared across days.

    Polished blanks (both Y and T) all subtract group 26 (yesterday's first polished blank), so
    today's polished blanks reveal the cross-day drift against yesterday rather than resetting to
    their own day's first scan. Unpolished has only yesterday, baseline is its first blank.
    Returns {(polished, day): baseline_group_id} for lookup convenience.
    """
    out = {}
    for pol in (False, True):
        yb = sorted(n for n, m in meta.items() if m["substance"] == "blank"
                    and m["disconnect"] == "none" and m["polished"] == pol and m["day"] == "Y")
        anchor = yb[0] if yb else None
        if anchor is None:
            continue
        for day in ("Y", "T"):
            has_day = any(m["substance"] == "blank" and m["disconnect"] == "none"
                          and m["polished"] == pol and m["day"] == day for m in meta.values())
            if has_day:
                out[(pol, day)] = anchor
    return out


# ---- background plan following the exact grouping (polish & day never crossed) ----

def _decade(ids: list[int]) -> tuple[list[int], list[int]]:
    """Split a sorted list of 10 into first-5 / last-5."""
    ids = sorted(ids)
    return ids[:5], ids[5:]


def background_plan(meta: dict | None = None) -> list[dict]:
    """Every (tag, sample_ids, blank_ids) background group. Polish and day are hard walls, never crossed.

    Yesterday, per polish state (10 blank + 10 sample each):
      - two 5+5 pairings (first-5 blanks -> first-5 samples, last-5 blanks -> last-5 samples)
      - one 10-whole pairing (all 10 blanks mean -> all 10 samples)
    Today (all polished, 20 blank + 20 sample):
      - four 5-blank means each -> matching 5-sample decile
      - two 10-blank means -> matching 10 samples
      - one 20-blank mean -> all 20 samples
    """
    if meta is None:
        meta = group_metadata()

    def blanks(pol, day):
        return sorted(n for n, m in meta.items() if m["substance"] == "blank"
                      and m["disconnect"] == "none" and m["polished"] == pol and m["day"] == day)

    def samples(pol, day):
        return sorted(n for n, m in meta.items() if m["substance"] == "sample"
                      and m["disconnect"] == "none" and m["polished"] == pol and m["day"] == day)

    plans = []
    # yesterday, both polish states, kept separate
    for pol, ptag in ((False, "unpolished"), (True, "polished")):
        b, s = blanks(pol, "Y"), samples(pol, "Y")
        if len(b) == 10 and len(s) == 10:
            b1, b2 = _decade(b)
            s1, s2 = _decade(s)
            plans.append({"tag": f"Y_{ptag}_5a", "blanks": b1, "samples": s1})
            plans.append({"tag": f"Y_{ptag}_5b", "blanks": b2, "samples": s2})
            plans.append({"tag": f"Y_{ptag}_10", "blanks": b, "samples": s})

    # today, all polished
    b, s = blanks(True, "T"), samples(True, "T")
    if len(b) == 20 and len(s) == 20:
        for q in range(4):
            bq = b[q * 5:(q + 1) * 5]
            sq = s[q * 5:(q + 1) * 5]
            plans.append({"tag": f"T_5_{q}", "blanks": bq, "samples": sq})
        plans.append({"tag": "T_10_a", "blanks": b[:10], "samples": s[:10]})
        plans.append({"tag": "T_10_b", "blanks": b[10:], "samples": s[10:]})
        plans.append({"tag": "T_20", "blanks": b, "samples": s})
    return plans


# background window for baseline slope: negative O2 region, well away from the +0.19V Fc peak
BASELINE_SLOPE_WIN = (-0.30, -0.10)


def _connected_groups(meta: dict, substance: str) -> dict:
    """{(polished, day): [group ids]} for a substance, disconnect=none, keeping only full CV groups."""
    out = {}
    for n, m in meta.items():
        if m["substance"] != substance or m["disconnect"] != "none":
            continue
        out.setdefault((m["polished"], m["day"]), []).append(n)
    return {k: sorted(v) for k, v in out.items()}


def export_pointwise(cv_root: str | Path, out_dir: str | Path, dt: float = 0.05,
                     meta: dict | None = None, scan: int = 1) -> dict:
    """Export per-E-point data the compressed tables cannot hold, for replicate-SD and slope plots.

    Two CSVs, one scan (default the second, cycle-stable scan):
      * blank_replicate_pointwise.csv - for each (polished, day) blank group, at every potential E:
        current mean and SD across the group's blank replicates. Drives the replicate-SD-vs-E curve.
      * baseline_slope.csv - for each blank in each group, the linear slope of current vs potential
        in the non-Faradaic background window (BASELINE_SLOPE_WIN). Drives slope variability by group.

    Replicate SD requires all replicates share the E grid; groups with mismatched point counts are
    truncated to the common length (reported in n_points).
    """
    cv_root, out_dir = Path(cv_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if meta is None:
        meta = group_metadata()

    groups = _connected_groups(meta, "blank")

    # --- replicate pointwise mean/SD ---
    rep_rows = []
    for (pol, day), ids in sorted(groups.items()):
        curves, E_ref = [], None
        for n in ids:
            p = cv_root / str(n) / "data.csv"
            if not p.exists():
                continue
            cv = load_cv(p)
            if scan not in cv["scans"]:
                continue
            E, i, _ = cv["scans"][scan]
            curves.append((E, i))
        if len(curves) < 2:
            continue
        nmin = min(len(i) for _, i in curves)
        E_ref = curves[0][0][:nmin]
        stack = np.vstack([i[:nmin] for _, i in curves])
        cur_mean = stack.mean(axis=0)
        cur_sd = stack.std(axis=0, ddof=1)
        for j in range(nmin):
            rep_rows.append({"polished": pol, "day": day, "n_replicates": len(curves),
                             "E": float(E_ref[j]),
                             "current_mean": float(cur_mean[j]),
                             "current_sd": float(cur_sd[j])})

    # --- baseline slope per blank ---
    slope_rows = []
    lo, hi = BASELINE_SLOPE_WIN
    for (pol, day), ids in sorted(groups.items()):
        for n in ids:
            p = cv_root / str(n) / "data.csv"
            if not p.exists():
                continue
            cv = load_cv(p)
            if scan not in cv["scans"]:
                continue
            E, i, _ = cv["scans"][scan]
            win = (E >= lo) & (E <= hi)
            if win.sum() < 5:
                continue
            slope, intercept = np.polyfit(E[win], i[win], 1)
            slope_rows.append({"polished": pol, "day": day, "blank": n,
                               "slope": float(slope), "intercept": float(intercept),
                               "n_points": int(win.sum())})

    for rows, name in ((rep_rows, "blank_replicate_pointwise.csv"),
                       (slope_rows, "baseline_slope.csv")):
        fields = sorted({k for r in rows for k in r})
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        LOG.info("Wrote %s (%d rows)", name, len(rows))
    return {"replicate_pointwise": rep_rows, "baseline_slope": slope_rows}


def build_diagnostic_tables(cv_root: str | Path, out_dir: str | Path, dt: float = 0.05,
                            meta: dict | None = None) -> dict:
    """Run the full unified diagnostic: three sample readings across every background plan, plus blank peak reports.

    Writes two CSVs: sample_readings (three methods vs FC_E0, per scan, per plan) and blank_peaks
    (every significant peak per blank by potential). Returns both as row lists.
    """
    cv_root, out_dir = Path(cv_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if meta is None:
        meta = group_metadata()

    sample_rows = []
    for plan in background_plan(meta):
        try:
            blank_mean = mean_blank_scans(plan["blanks"], cv_root)
        except (FileNotFoundError, ValueError) as e:
            LOG.warning("plan %s: blank mean failed (%s)", plan["tag"], e)
            continue
        for smp in plan["samples"]:
            ps = cv_root / str(smp) / "data.csv"
            if not ps.exists():
                continue
            cs = load_cv(ps)
            for k in sorted(cs["scans"]):
                if k not in blank_mean:
                    continue
                E_s, i_s, _ = cs["scans"][k]
                row = {"plan": plan["tag"], "sample": smp, "scan": k, "n_blanks": len(plan["blanks"])}
                row.update(sample_scan_three_readings(E_s, i_s, blank_mean[k], dt))
                sample_rows.append(row)

    # blank residual peaks: every connected blank MINUS its batch baseline (first blank of same polish+day)
    blank_rows = []
    baselines = _baseline_ids(meta)
    base_cache = {}
    for n, m in meta.items():
        if m["substance"] != "blank" or m["disconnect"] != "none":
            continue
        pb = cv_root / str(n) / "data.csv"
        if not pb.exists():
            continue
        base_id = baselines.get((m["polished"], m["day"]))
        if base_id is None:
            continue
        if base_id not in base_cache:
            bpath = cv_root / str(base_id) / "data.csv"
            base_cache[base_id] = load_cv(bpath) if bpath.exists() else None
        base_cv = base_cache[base_id]
        if base_cv is None:
            continue
        cb = load_cv(pb)
        for k in sorted(cb["scans"]):
            if k not in base_cv["scans"]:
                continue
            E, i, _ = cb["scans"][k]
            is_baseline = (n == base_id)
            res = blank_residual_peaks(E, i, (base_cv["scans"][k][0], base_cv["scans"][k][1]), dt)
            glob = {"resid_rms": res["resid_rms"], "resid_neg_mean": res["resid_neg_mean"],
                    "resid_max_abs": res["resid_max_abs"]}
            if res["peaks"]:
                for pk in res["peaks"]:
                    blank_rows.append({"blank": n, "baseline": base_id, "is_baseline": is_baseline,
                                       "scan": k, "polished": m["polished"], "day": m["day"],
                                       **glob, **pk})
            else:
                blank_rows.append({"blank": n, "baseline": base_id, "is_baseline": is_baseline,
                                   "scan": k, "polished": m["polished"], "day": m["day"], **glob,
                                   "branch": "none", "E_peak": np.nan, "sd_height": np.nan,
                                   "peak_over_noise": np.nan})

    for rows, name in ((sample_rows, "table4_sample_readings.csv"),
                       (blank_rows, "table5_blank_peaks.csv")):
        fields = sorted({kk for r in rows for kk in r})
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        LOG.info("Wrote %s (%d rows)", name, len(rows))
    return {"sample_readings": sample_rows, "blank_peaks": blank_rows}