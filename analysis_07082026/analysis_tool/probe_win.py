import numpy as np, Project.electrochem.cv_core as cc
from scipy.signal import find_peaks
import diag_fit
sample = diag_fit.load_cv('../test_2/test_iso_4/Cv2/data.csv')
blank = diag_fit.load_cv('../test_2/test_iso_4/Cv1/data.csv')
resid = cc.subtract_blank(sample, blank)
E, i, t = resid[0]
sig = cc.noise_mad(i)
pk, props = find_peaks(i, prominence=cc.PROM_K*sig, distance=cc.FIND_DISTANCE)
print('scan0 正峰:')
for p, l, r in zip(pk, props['left_bases'], props['right_bases']):
    print(f'  peak_idx={p} E_peak={E[p]:.3f} | left_base={l} E={E[l]:.3f} | right_base={r} E={E[r]:.3f}')
