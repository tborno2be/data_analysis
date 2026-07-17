"""CV core: RL semiderivative (cross/per-scan), bipolar coarse nomination, three-line Pearson fitting."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, fftconvolve
from scipy.special import gamma as sp_gamma
from lmfit.models import LinearModel, Pearson4Model

SD_SCALE = 1e6
NOISE_K = 5.0
HALFWIN_S = 2.0
GUARD_S = 0.5
FIND_DISTANCE = 15
CENTER_BOX_S = 0.5
EDGE_FRAC = 0.10
DBIC_ACCEPT = -2.0
DBIC_STRONG = -10.0
MAX_COMBO = 4
LINES = ("frozen", "frozen_locks", "joint_locks")


def fracdiff(i, dt, alpha):
    """RL (Oldham-Spanier) differintegral of order alpha; negative alpha integrates."""
    n = len(i)
    j = np.arange(n, dtype=float)
    w = np.empty(n)
    w[0] = 1.0 / sp_gamma(2.0 - alpha)
    w[1:] = ((j[1:] + 1) ** (1 - alpha) - 2 * j[1:] ** (1 - alpha)
             + (j[1:] - 1) ** (1 - alpha)) / sp_gamma(2.0 - alpha)
    out = dt ** (-alpha) * fftconvolve(i, w)[:n]
    out[0] = 0.0
    return out


def succ_sigma(x):
    """Noise sigma from successive differences via MAD."""
    d = np.diff(x)
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2))


def transform(i_scans, dt, order=0.5, mode="continuous"):
    """Semiderivative of the scan list on one concatenated axis; returns (sd, slices, guard)."""
    lens = [len(i) for i in i_scans]
    edges = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    slices = [slice(edges[j], edges[j + 1]) for j in range(len(i_scans))]
    i_all = np.concatenate(i_scans)
    guard = np.zeros(len(i_all), bool)
    g = int(GUARD_S / dt)
    if mode == "continuous":
        sd = fracdiff(i_all * SD_SCALE, dt, order)
        guard[:g] = True
    else:
        sd = np.empty(len(i_all))
        for sl in slices:
            sd[sl] = fracdiff(i_all[sl] * SD_SCALE, dt, order)
            guard[sl.start:sl.start + g] = True
    return sd, slices, guard


def nominate(sd, slices, guard, E_all, k=NOISE_K):
    cands = []
    for s_idx, sl in enumerate(slices):
        y = sd[sl]
        sig = succ_sigma(y)
        for pol in (1, -1):
            pk, props = find_peaks(pol * y, height=k*sig, prominence=sig, distance=FIND_DISTANCE)
            for p, prom in zip(pk, props["prominences"]):
                gi = sl.start + int(p)
                if not guard[gi]:
                    cands.append({"scan": s_idx, "idx": gi, "polarity": pol,
                                  "prom": float(prom), "sigma": sig, "E": float(E_all[gi])})
    # collapse by electrode potential (same physical peak across scans), keep strongest
    seen = {}
    for c in sorted(cands, key=lambda x: -x["prom"]):
        key = (round(c["E"] / 0.01), c["polarity"])   # 10 mV bins
        seen.setdefault(key, c)
    return list(seen.values())

    return cands


def clusters(cands, dt, halfwin=HALFWIN_S):
    """Merge candidates whose fit windows overlap on the global axis."""
    if not cands:
        return []
    cands = sorted(cands, key=lambda c: c["idx"])
    w = int(halfwin / dt)
    out, cur = [], [cands[0]]
    for c in cands[1:]:
        if c["idx"] - cur[-1]["idx"] <= 2 * w:
            cur.append(c)
        else:
            out.append(cur)
            cur = [c]
    out.append(cur)
    return out


def _bic(rss, n, k):
    """Bayesian information criterion."""
    return n * np.log(max(rss, 1e-300) / n) + k * np.log(n)


def _fwhm(sigma, expon):
    """Pearson IV effective full width at half maximum."""
    return 2.0 * sigma * np.sqrt(2 ** (1.0 / expon) - 1.0)


def _make_model(subset, t, y, sd, lock, joint):
    """Composite Linear+Pearson model over an explicit candidate subset; p{j}_ maps to subset[j]."""
    model, pars = None, None
    if joint:
        model = LinearModel(prefix="bg_")
        pars = model.guess(np.r_[y[:3], y[-3:]], x=np.r_[t[:3], t[-3:]])
    for j, c in enumerate(subset):
        pk = Pearson4Model(prefix=f"p{j}_")
        model = pk if model is None else model + pk
        p = pk.make_params()
        t0 = c["t0"]
        p[f"p{j}_center"].set(value=t0, min=t0 - CENTER_BOX_S, max=t0 + CENTER_BOX_S)
        p[f"p{j}_sigma"].set(value=0.3, min=1e-3)
        p[f"p{j}_expon"].set(value=1.5, min=0.6)
        p[f"p{j}_skew"].set(value=0.0)
        p[f"p{j}_amplitude"].set(value=c["polarity"] * abs(sd[c["idx"]]) * 0.5)
        if lock:
            key = f"p{j}_amplitude"
            p[key].set(min=0.0) if c["polarity"] > 0 else p[key].set(max=0.0)
        if pars is None:
            pars = p
        else:
            pars.update(p)
    return model, pars


def _fit_subset(subset, t, y, sd, dt, lock, joint, extra, n_pts):
    """Fit one candidate subset; returns (result_or_None, bic)."""
    if not subset:
        if joint:
            model = LinearModel(prefix="bg_")
            pars = model.guess(np.r_[y[:3], y[-3:]], x=np.r_[t[:3], t[-3:]])
            res = model.fit(y, pars, x=t)
            return res, _bic(float(np.sum(res.residual ** 2)), n_pts, res.nvarys)
        return None, _bic(float(np.sum(y ** 2)), n_pts, extra)
    model, pars = _make_model(subset, t, y, sd, lock, joint)
    try:
        res = model.fit(y, pars, x=t)
    except Exception:
        return None, np.inf
    return res, _bic(float(np.sum(res.residual ** 2)), n_pts, res.nvarys + extra)


def fit_cluster(cluster, sd, t_all, slices, dt, line, mode, order=0.5):
    """BIC forward selection over nested Pearson ladders; returns (components, status, margin)."""
    lock = line.endswith("locks")
    joint = line.startswith("joint")
    cands = sorted(cluster, key=lambda c: -c["prom"])
    w = int(HALFWIN_S / dt)
    lo = max(min(c["idx"] for c in cands) - w, 0)
    hi = min(max(c["idx"] for c in cands) + w, len(sd))
    if mode == "per_scan":
        sl = slices[cands[0]["scan"]]
        lo, hi = max(lo, sl.start), min(hi, sl.stop)
    t, y = t_all[lo:hi], sd[lo:hi].copy()
    if len(t) < 10:
        return [], "fit_failed", np.nan
    extra = 0
    if not joint:
        slope, icpt = np.polyfit(np.r_[t[:3], t[-3:]], np.r_[y[:3], y[-3:]], 1)
        y = y - (slope * t + icpt)
        extra = 2
    for c in cands:
        c["t0"] = t_all[c["idx"]]
    n_pts = len(y)
    best_subset, best_res, best_bic, margin = [], None, None, np.nan

    if len(cands) <= MAX_COMBO:
        # exhaustive: every subset by size, so a spurious top-prominence candidate can't bias the ladder
        from itertools import combinations
        by_size = {}
        for k in range(len(cands) + 1):
            best_k = None
            for sub in combinations(cands, k):
                res, bic = _fit_subset(list(sub), t, y, sd, dt, lock, joint, extra, n_pts)
                if best_k is None or bic < best_k[1]:
                    best_k = (list(sub), bic, res)
            by_size[k] = best_k
        prev_bic = None
        for k in range(len(cands) + 1):
            sub, bic, res = by_size[k]
            if prev_bic is None or bic - prev_bic <= DBIC_ACCEPT:
                if prev_bic is not None:
                    margin = bic - prev_bic
                best_subset, best_res, best_bic, prev_bic = sub, res, bic, bic
            else:
                break
    else:
        # forward selection by prominence for large clusters (combinatorial guard)
        prev_bic = None
        for n in range(0, len(cands) + 1):
            res, bic = _fit_subset(cands[:n], t, y, sd, dt, lock, joint, extra, n_pts)
            if prev_bic is None or bic - prev_bic <= DBIC_ACCEPT:
                if prev_bic is not None:
                    margin = bic - prev_bic
                best_subset, best_res, prev_bic = cands[:n], res, bic
            else:
                break

    if not best_subset or best_res is None:
        return [], "no_peak", margin
    span = t_all[hi - 1] - t_all[lo]
    parts = best_res.eval_components(x=t)
    comps = []
    for j, src in enumerate(best_subset):
        p = best_res.params
        height = float(p[f"p{j}_height"].value)
        fwhm = _fwhm(float(p[f"p{j}_sigma"].value), float(p[f"p{j}_expon"].value))
        center = float(p[f"p{j}_center"].value)
        sig_j = src["sigma"]
        pon = abs(height) / sig_j if sig_j > 0 else np.nan
        ok = (pon >= NOISE_K and 3 * dt <= fwhm <= span
              and min(center - t_all[lo], t_all[hi - 1] - center) >= EDGE_FRAC * span)
        lsv = fracdiff(parts[f"p{j}_"], dt, -order) / SD_SCALE
        ip = float(np.sign(height)) * float(lsv.max() - lsv.min())
        comps.append({"center": center, "polarity": 1 if height >= 0 else -1,
                      "height": height, "fwhm": fwhm, "pon": pon, "ip": ip, "qc": ok,
                      "scan": src["scan"], "source_idx": src["idx"],
                      "source_sigma": sig_j, "source_prom": src["prom"],
                      "prefix": f"p{j}_", "result": best_res})
    kept = [c for c in comps if c["qc"]]
    if not kept:
        return [], "rejected_noise", margin
    status = ("single_resolved" if len(kept) == 1
              else "multi_resolved" if margin <= DBIC_STRONG
              else "unresolved_composite")
    return kept, status, margin