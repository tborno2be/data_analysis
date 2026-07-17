"""Verify the CA-side disconnect thresholds reproduce Book1 labels 100%.
Run after ca_disconnect_features.py. Usage: python ca_verify.py ca_disconnect_feats.csv"""

from __future__ import annotations
import sys
import pandas as pd


def classify_ca(max_abs_i, sigma_noise, flips):
    """CA-side electrode-disconnect state. Order matters: reference first (current blows up),
    then working (open circuit -> pA noise floor + sign thrash), then counter (sign thrash)."""
    if max_abs_i > 5e-5:                       # reference: potentiostat loses control, current 100x up
        return "reference"
    if sigma_noise < 1e-9 and flips > 0.1:     # working: open, noise collapses to pA and sign flips
        return "working"
    if flips > 0.5:                            # counter: pure-noise trace, sign flips constantly
        return "counter"
    return "none"


def main(feats):
    d = pd.read_csv(feats)
    ok = 0
    conf = {}
    for _, r in d.iterrows():
        pred = classify_ca(r.max_abs_i, r.sigma_noise, r.sign_flips)
        truth = r.disconnect
        conf[(truth, pred)] = conf.get((truth, pred), 0) + 1
        if pred == truth:
            ok += 1
        else:
            print(f"  MISS group{int(r.group)}: truth={truth} pred={pred} "
                  f"| max_abs_i={r.max_abs_i:.2e} sigma={r.sigma_noise:.2e} flips={r.sign_flips:.3f}")
    print(f"\nCA disconnect classifier: {ok}/{len(d)} = {ok/len(d)*100:.1f}%")
    print("\nconfusion (truth -> pred): count")
    for (t, p), n in sorted(conf.items()):
        mark = "" if t == p else "  <-- ERROR"
        print(f"  {t:10} -> {p:10}: {n}{mark}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "ca_disconnect_feats.csv")