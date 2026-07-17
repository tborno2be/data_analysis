"""CV peak core (shape-agnostic). Pipeline:
  0 subtract_blank -> 1 nominate_current -> 2 transform(per/cross) -> 3 confirm_sd
  -> 4 cluster(by E) -> 5 fit_in_cluster(3 locks) -> 6 read_params(semi-integral).
Peak shape (Pearson/Gaussian) is isolated behind PeakModel; the rest is shape-agnostic.
Cross-scan: one semiderivative over the concatenated series (no branch cutting); the series is
never physically cut - segments are only READ (for local noise, for windows) then discarded."""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, fftconvolve
from scipy.special import gamma as sp_gamma
from lmfit.models import LinearModel
from cv_peak import PeakModel, PearsonPeak, GaussianPeak

SD_SCALE = 1e6
PROM_K = 3.0          # current-domain prominence gate (x noise_current)
CONFIRM_K = 5.0       # sd-domain peak-height gate (x noise_sd)
GUARD_S = 0.5         # guard half-width around series start and every vertex
FIND_DISTANCE = 15
E_CLUSTER_V = 0.02    # merge same-polarity candidates within this potential
WIN_S = 1.5           # fit window half-width (seconds)
CENTER_BOX_S = 0.5    # lock 3a: center may move at most this far
MAX_PEAKS = 3         # CV has few peaks; BIC normally stops at 1 (blowup guard)
DBIC_ACCEPT = -6.0
DBIC_STRONG = -10.0


# ---------------------------------------------------------------- primitives
def fracdiff(y, dt, alpha):
    """RL (Oldham-Spanier) differintegral of order alpha; alpha<0 integrates."""
    n = len(y)
    j = np.arange(n, dtype=float)
    w = np.empty(n)
    w[0] = 1.0 / sp_gamma(2.0 - alpha)
    w[1:] = ((j[1:] + 1) ** (1 - alpha) - 2 * j[1:] ** (1 - alpha)
             + (j[1:] - 1) ** (1 - alpha)) / sp_gamma(2.0 - alpha)
    out = dt ** (-alpha) * fftconvolve(y, w)[:n]
    out[0] = 0.0
    return out


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def noise_mad(x):
    """Successive-difference MAD noise estimate."""
    if len(x) < 3:
        return np.nan
    d = np.diff(x)
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / np.sqrt(2))


def _vertices(E):
    """Indices where scan direction reverses (dE/dt changes sign) within one scan."""
    dE = np.diff(E)
    s = np.sign(dE)
    return list(np.where(np.diff(s) != 0)[0] + 1)


# ---------------------------------------------------------------- 0. blank
def subtract_blank(sample, blank):
    """Per-scan sample-minus-blank current; returns {scan:(E,i,t)}. Raises on mismatch."""
    keys = sorted(set(sample) & set(blank))
    if not keys:
        raise ValueError("no common scans between sample and blank")
    out = {}
    for k in keys:
        E, i_s, t = sample[k]
        _, i_b, _ = blank[k]
        if len(i_s) != len(i_b):
            raise ValueError(f"scan {k}: sample {len(i_s)} vs blank {len(i_b)} points")
        out[k] = (E, i_s - i_b, t)
    return out


# ---------------------------------------------------------------- 1. nominate
def nominate_current(resid, dt):
    """Current-domain candidates per scan: find_peaks on +/-i, prom = PROM_K * noise_current(scan).
    The window is anchored on the peak index and sized by the peak's half-prominence width in INDEX
    space (widths from find_peaks), NOT by base potentials -- on a reversing sweep the same potential
    recurs, so base potentials can land on the wrong branch and misplace the window. E_lo/E_hi are
    derived from the index window purely as labels. Returns
    [{scan, idx, E_center, E_lo, E_hi, polarity, prom}]."""
    cands = []
    for k in sorted(resid):
        E, i, _ = resid[k]
        n = len(i)
        sig_i = noise_mad(i)
        if not np.isfinite(sig_i) or sig_i <= 0:
            continue
        for pol in (1, -1):
            pk, props = find_peaks(pol * i, prominence=PROM_K * sig_i, distance=FIND_DISTANCE)
            verts = _vertices(E)                          # reversal points within this scan
            for p, l, r, prom in zip(pk, props["left_bases"], props["right_bases"],
                                     props["prominences"]):
                lo, hi = int(l), int(r)                   # base indices: the peak's true extent
                for v in verts:                           # clip at the reversal (never cross branches)
                    if lo < v <= p:
                        lo = v + 1
                    elif p <= v < hi:
                        hi = v
                cands.append({"scan": k, "idx": int(p), "E_center": float(E[p]),
                              "E_lo": float(min(E[lo], E[hi])),
                              "E_hi": float(max(E[lo], E[hi])),
                              "lo": lo, "hi": hi,
                              "polarity": int(pol), "prom": float(prom)})
    return cands


# ---------------------------------------------------------------- 2. transform
def transform(resid, dt, order, mode):
    """Semiderivative. cross_scan: one fracdiff over the concatenated series (no branch cutting);
    per_scan: fracdiff each scan independently. guard covers the series start and every vertex +/-GUARD_S.
    Returns sd, slices{scan:slice}, guard(bool), t_all, E_all, vertices(global idx list)."""
    keys = sorted(resid)
    lens = [len(resid[k][1]) for k in keys]
    edges = np.concatenate([[0], np.cumsum(lens)]).astype(int)
    slices = {k: slice(edges[j], edges[j + 1]) for j, k in enumerate(keys)}
    i_all = np.concatenate([resid[k][1] for k in keys])
    E_all = np.concatenate([resid[k][0] for k in keys])
    guard = np.zeros(len(i_all), bool)
    g = int(GUARD_S / dt)

    if mode == "cross_scan":
        sd = fracdiff(i_all * SD_SCALE, dt, order)
        guard[:g] = True
    elif mode == "per_scan":
        sd = np.empty(len(i_all))
        for sl in slices.values():
            sd[sl] = fracdiff(i_all[sl] * SD_SCALE, dt, order)
            guard[sl.start:sl.start + g] = True
    else:
        raise ValueError(f"unknown mode {mode}")

    # global vertex indices; guard each vertex +/- g
    vertices = []
    for k in keys:
        sl = slices[k]
        for v in _vertices(E_all[sl]):
            gv = sl.start + v
            vertices.append(gv)
            guard[max(gv - g, 0):min(gv + g, len(guard))] = True

    t_all = np.arange(len(sd)) * dt
    return sd, slices, guard, t_all, E_all, sorted(vertices)


# ---------------------------------------------------------------- 3. confirm
def confirm_sd(sd, cands, slices, E_all, guard):
    """Confirm each current candidate in the sd domain: within its INDEX window (the time anchor from
    nominate, unique on a reversing sweep -- unlike a potential window, which recurs), the sd peak
    height must exceed CONFIRM_K * noise_sd, estimated LOCALLY on that scan's segment (a segment is
    only READ; the sd array is never cut). guard rejects vertex/edge."""
    out = []
    for c in cands:
        sl = slices[c["scan"]]
        seg_sd = sd[sl]
        sig_sd = noise_mad(seg_sd)
        if not np.isfinite(sig_sd) or sig_sd <= 0:
            continue
        lo = sl.start + c["lo"]           # candidate's index window, mapped into the global sd
        hi = sl.start + c["hi"]
        lo, hi = max(lo, sl.start), min(hi, sl.stop)
        if hi - lo < 3:
            continue
        y = c["polarity"] * sd[lo:hi]
        loc = int(np.argmax(y))
        gi = lo + loc
        if guard[gi] or y[loc] < CONFIRM_K * sig_sd:
            continue
        out.append({"scan": c["scan"], "idx": gi, "polarity": c["polarity"],
                    "prom": c["prom"], "sigma_sd": sig_sd, "E_center": c["E_center"],
                    "E_lo": c["E_lo"], "E_hi": c["E_hi"]})
    return out


# ---------------------------------------------------------------- 4. cluster
def cluster(cands):
    """Merge same-polarity candidates whose E_center is within E_CLUSTER_V into one cluster
    (= all cross-scan detections of one physical peak). Positive/negative never merge."""
    out = []
    for pol in (1, -1):
        grp = sorted([c for c in cands if c["polarity"] == pol], key=lambda c: c["E_center"])
        cur = []
        for c in grp:
            if cur and c["E_center"] - cur[-1]["E_center"] > E_CLUSTER_V:
                out.append(cur)
                cur = []
            cur.append(c)
        if cur:
            out.append(cur)
    return out


# ---------------------------------------------------------------- 5. fit in cluster
OV_MAX = 0.5          # reject peak-add if smaller peak has >half its area buried in another
FWHM_MIN_ABS = 0.15   # reject peak-add if any component narrower than this (seconds); anti-fragment


def _legal_multi(res, t, pm, dt):
    """A multi-peak fit is legal only if no pair overlaps beyond OV_MAX and no peak is a fragment.
    Blocks (1) splitting one blunt peak into overlapping peaks, (2) sprinkling narrow peaks on noise."""
    prefs = sorted({k.rsplit("_", 1)[0] + "_" for k in res.params if k.endswith("center")})
    comps = res.eval_components(x=t)
    curves = [np.abs(comps[p]) for p in prefs]
    for p, c in zip(prefs, curves):
        if pm.fwhm(res.params, p) < FWHM_MIN_ABS:          # fragment: too narrow
            return False
    for a in range(len(curves)):
        for b in range(a + 1, len(curves)):
            ca, cb = curves[a], curves[b]
            ai, aj = _trapz(ca, t), _trapz(cb, t)
            denom = min(ai, aj)
            if denom > 0 and _trapz(np.minimum(ca, cb), t) / denom > OV_MAX:
                return False                                # split: too much overlap
    return True


def _bic(rss, n, k):
    return n * np.log(max(rss, 1e-300) / n) + k * np.log(n)


def _window(idx, sd_len, dt, mode, sl, vertices):
    """Open a fit window around idx; cross uses global bounds, per uses the scan slice.
    Never span a vertex: clip the window to the vertex on the side between center and vertex."""
    w = int(WIN_S / dt)
    if mode == "per_scan":
        lo, hi = max(idx - w, sl.start), min(idx + w, sl.stop)
    else:
        lo, hi = max(idx - w, 0), min(idx + w, sd_len)
    for v in vertices:
        if lo < v <= idx:        # vertex on the left of center
            lo = max(lo, v + 1)
        elif idx <= v < hi:      # vertex on the right of center
            hi = min(hi, v)
    return lo, hi


def _fit_one_detection(cand, sd, t_all, sl, dt, order, bg, pm, mode, vertices):
    """Fit the window around ONE detection (one scan). 3 locks. Returns (peaks, status, margin)."""
    lo, hi = _window(cand["idx"], len(sd), dt, mode, sl, vertices)
    t, y = t_all[lo:hi], sd[lo:hi].copy()
    if len(t) < 10:
        return [], "fit_failed", np.nan
    joint = (bg == "joint")
    extra = 0
    if not joint:                                       # frozen background: edge line, subtract
        slope, icpt = np.polyfit(np.r_[t[:3], t[-3:]], np.r_[y[:3], y[-3:]], 1)
        y = y - (slope * t + icpt)
        extra = 2
    tpk = t_all[cand["idx"]]
    n_pts = len(y)
    best_n, best_res, prev_bic, margin = 0, None, None, np.nan
    for n in range(1, MAX_PEAKS + 1):                    # lock 2: BIC ladder, greedy single->multi
        model, pars = None, None
        if joint:
            model = LinearModel(prefix="bg_")
            pars = model.guess(np.r_[y[:3], y[-3:]], x=np.r_[t[:3], t[-3:]])
        for jj in range(n):
            pref = f"p{jj}_"
            pm_model, pm_pars = pm.build(pref, tpk, cand["polarity"], sd[cand["idx"]])  # lock 1: sign
            model = pm_model if model is None else model + pm_model
            pars = pm_pars if pars is None else (pars.update(pm_pars) or pars)
        try:
            res = model.fit(y, pars, x=t)
        except Exception:
            break
        bic = _bic(float(np.sum(res.residual ** 2)), n_pts, res.nvarys + extra)
        if prev_bic is None or bic - prev_bic <= DBIC_ACCEPT:
            # BIC supports adding this peak; now check the add is LEGAL (not a split of one peak)
            if n >= 2 and not _legal_multi(res, t, pm, dt):
                break                                    # illegal split -> reject, keep previous count
            if prev_bic is not None:
                margin = bic - prev_bic
            best_n, best_res, prev_bic = n, res, bic
        else:
            break
    if best_n == 0 or best_res is None:
        return [], "no_peak", margin
    span = t[-1] - t[0]
    rss = float(np.sum(best_res.residual ** 2))
    peaks = []
    for jj in range(best_n):
        pref = f"p{jj}_"
        p = best_res.params
        center = float(p[f"{pref}center"].value)
        fw = pm.fwhm(p, pref)
        # lock 3: center already box-bounded in build; FWHM upper bound = window span
        fwhm_ok = bool(3 * dt <= fw <= span)
        center_at_box = bool(abs(center - tpk) >= CENTER_BOX_S - dt)   # center hit its bound
        peaks.append({"scan": cand["scan"], "center_t": center, "polarity": cand["polarity"],
                      "fwhm": fw, "shape_diag": pm.shape_diag(p, pref),
                      "height": pm.height(p, pref), "prefix": pref, "result": best_res,
                      "sd_rss": rss, "sd_redchi": float(best_res.redchi),
                      "qc": fwhm_ok and not center_at_box,
                      "edge_truncated": center_at_box, "win_lo": lo, "win_hi": hi})
    status = "single" if best_n == 1 else ("multi" if margin <= DBIC_STRONG else "unresolved")
    return peaks, status, margin


# ---------------------------------------------------------------- confidence
# Peak validity as a product of independent [0,1] discount factors (one-vote veto, no weights).
# Transition points are parameters (calibrated on real Fc peaks vs blank no-peak candidates), not gates.
CONF = {
    "pon_lo": 3.0, "pon_hi": 8.0,        # f_snr ramps from 0 at pon_lo to 1 at pon_hi
    "ov_lo": 0.3, "ov_hi": 0.5,          # f_overlap: <lo ->1, lo..hi linear, >hi ->0
    "fwhm_min": 0.15, "fwhm_soft": 0.4,  # f_width: <min ->0, min..soft ramp, >soft ->1 (absolute FWHM, s)
    "edge_penalty": 0.3,                 # f_edge multiplier when edge_truncated
}


def _ramp(x, lo, hi):
    """0 below lo, 1 above hi, linear between (hi>lo)."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _overlap_frac(pi, pj, t):
    """int min(pi,pj) / min(area_i,area_j): fraction of the smaller peak buried in the other."""
    ai, aj = _trapz(np.abs(pi), t), _trapz(np.abs(pj), t)
    denom = min(ai, aj)
    if denom <= 0:
        return 0.0
    return float(_trapz(np.minimum(np.abs(pi), np.abs(pj)), t) / denom)


def peak_confidence(pk, pon, max_overlap, cf=CONF):
    """Four model-agnostic [0,1] factors, multiplied. Works for any peak shape (Pearson/Gaussian):
    every factor is a geometric/statistical quantity, none reads a shape-specific parameter.
    skew is NOT used here (it measures single-peak asymmetry, not peak reality, and is Pearson-only)."""
    f_snr = _ramp(pon, cf["pon_lo"], cf["pon_hi"])
    f_overlap = 1.0 - _ramp(max_overlap, cf["ov_lo"], cf["ov_hi"])       # 1 when isolated, 0 when buried
    f_width = _ramp(pk["fwhm"], cf["fwhm_min"], cf["fwhm_soft"])
    f_edge = cf["edge_penalty"] if pk["edge_truncated"] else 1.0
    conf = f_snr * f_overlap * f_width * f_edge
    return {"confidence": float(conf), "f_snr": f_snr, "f_overlap": f_overlap,
            "f_width": f_width, "f_edge": f_edge}


def _score_detection(peaks, sd, t_all, dt, cf):
    """Attach pon, per-pair max overlap, and confidence to each peak in one detection's fit."""
    for pk in peaks:
        res, pref = pk["result"], pk["prefix"]
        xw = t_all[pk["win_lo"]:pk["win_hi"]]
        seg = sd[pk["win_lo"]:pk["win_hi"]]
        resid_seg = seg - res.eval(x=xw)       # data minus full model = noise (peak removed)
        sig = noise_mad(resid_seg)
        pk["pon"] = float(abs(pk["height"]) / sig) if sig > 0 else np.nan
    # pairwise overlap within this detection (multi-peak only); single peak -> 0
    t = t_all[peaks[0]["win_lo"]:peaks[0]["win_hi"]] if peaks else None
    for pk in peaks:
        mx = 0.0
        for other in peaks:
            if other is pk:
                continue
            ci = pk["result"].eval_components(x=t)[pk["prefix"]]
            cj = other["result"].eval_components(x=t)[other["prefix"]]
            mx = max(mx, _overlap_frac(ci, cj, t))
        pk["max_overlap"] = mx
        pk.update(peak_confidence(pk, pk["pon"], mx, cf))


def fit_in_cluster(clust, sd, slices, t_all, dt, order, bg, mode, pm, vertices, cf=CONF):
    """Fit each scan's detection independently; score confidence; aggregate across scans.
    Returns {peaks:[per-detection], agg:{...}}."""
    per = []
    n_scans_in_clust = len({c["scan"] for c in clust})
    for cand in clust:
        sl = slices[cand["scan"]]
        peaks, status, margin = _fit_one_detection(cand, sd, t_all, sl, dt, order, bg, pm, mode, vertices)
        if peaks:
            _score_detection(peaks, sd, t_all, dt, cf)
        for pk in peaks:
            pk["cluster_status"] = status
            pk["bic_margin"] = margin
            per.append(pk)
    good = [pk for pk in per if pk["qc"]]
    agg = {"n_detections": len(clust), "n_fitted": len(per),
           "n_scans_detected": n_scans_in_clust}
    if good:
        cts = np.array([pk["center_t"] for pk in good])
        med_ct = float(np.median(cts))
        mad_ct = float(np.median(np.abs(cts - med_ct)))
        outliers = [pk["scan"] for pk in good if abs(pk["center_t"] - med_ct) > 3 * max(mad_ct, 1e-9)]
        agg.update({"median_center_t": med_ct, "mad_center_t": mad_ct,
                    "polarity": good[0]["polarity"], "outlier_scans": outliers,
                    "n_good": len(good)})
    return {"peaks": per, "agg": agg}


# ---------------------------------------------------------------- 6. read params
def read_params(fit_result, order, dt, t_all, E_all):
    """Semi-integral each fitted peak back to the current domain; ip = step height (max-min).
    Attach Ep and ip per detection; add cluster-level median Ep/ip and detection_rate (needs n_scans)."""
    peaks = fit_result["peaks"]
    rows = []
    for pk in peaks:
        res, pref = pk["result"], pk["prefix"]
        comp = res.eval_components(x=t_all)[pref]
        lsv = fracdiff(comp, dt, -order) / SD_SCALE
        ip = float(np.sign(pk["height"])) * float(lsv.max() - lsv.min())
        Ep = float(np.interp(pk["center_t"], t_all, E_all))
        rows.append({"scan": pk["scan"], "polarity": pk["polarity"], "Ep": Ep, "ip": ip,
                     "fwhm": pk["fwhm"], "shape_diag": pk["shape_diag"], "qc": pk["qc"],
                     "edge_truncated": pk["edge_truncated"], "sd_rss": pk["sd_rss"],
                     "sd_redchi": pk["sd_redchi"], "pon": pk.get("pon", np.nan),
                     "max_overlap": pk.get("max_overlap", 0.0),
                     "confidence": pk.get("confidence", np.nan),
                     "f_snr": pk.get("f_snr", np.nan), "f_overlap": pk.get("f_overlap", np.nan),
                     "f_width": pk.get("f_width", np.nan),
                     "f_edge": pk.get("f_edge", np.nan),
                     "cluster_status": pk["cluster_status"], "bic_margin": pk["bic_margin"]})
    good = [r for r in rows if r["qc"]]
    agg = dict(fit_result["agg"])
    if good:
        agg["median_Ep"] = float(np.median([r["Ep"] for r in good]))
        agg["median_ip"] = float(np.median([r["ip"] for r in good]))
        agg["mad_ip"] = float(np.median(np.abs(np.array([r["ip"] for r in good])
                                               - np.median([r["ip"] for r in good]))))
    return {"detections": rows, "agg": agg}