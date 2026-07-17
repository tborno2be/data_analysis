# diag_real.py — 放 analysis_tool，跑: python diag_real.py <你这个CV的data.csv路径>
import sys, numpy as np
import cv_core as cc, cv_peak

path = sys.argv[1]
rows = []
for ln in open(path):
    ln = ln.strip()
    if ',' not in ln: continue
    try:
        a, b = ln.split(','); rows.append((float(a), float(b)))
    except: pass
E = np.array([r[0] for r in rows]); I = np.array([r[1] for r in rows])
dt = 0.001 / 0.1                       # 1 mV step / 0.1 V/s = 0.01 s；按你实际改
t = np.arange(len(E)) * dt
resid = cc.subtract_blank({0:(E,I,t)}, {0:(E,np.zeros_like(I),t)})   # 无blank,零背景
sd, slices, guard, t_all, E_all, verts = cc.transform(resid, dt, 0.5, "cross_scan")
cands = cc.nominate_current(resid, dt)
conf = cc.confirm_sd(sd, cands, slices, E_all, guard)
clusters = cc.cluster(conf)
print(f"nominate {len(cands)}, confirm {len(conf)}, clusters {len(clusters)}")
pm = cv_peak.PearsonPeak()
for cl in clusters:
    fr = cc.fit_in_cluster(cl, sd, slices, t_all, dt, 0.5, "frozen", "cross_scan", pm, verts)
    rp = cc.read_params(fr, 0.5, dt, t_all, E_all)
    for d in rp["detections"]:
        print(f"  pol={d['polarity']:+d} Ep={d['Ep']:.3f} ip={d['ip']:.2e} "
              f"fwhm={d['fwhm']:.2f} skew={d['shape_diag']:.2f} "
              f"status={d['cluster_status']} qc={d['qc']}")