"""Compute table2/table3 CV features for the test_iso Cv files with unchanged definitions."""
import csv
import sys
from pathlib import Path

from analysis import (load_cv, cv_scan_peaks, cv_scan_background,
                      cv_interscan_drift, feature_vector)

META = {"dataset": "test_iso_2", "substance": "sample", "day": "today",
        "polished": "", "disconnect": "none"}


def rows_for(path, group):
    """Table2 and table3 rows for one CV file, same loop as build_cv_tables."""
    cv = load_cv(path)
    prow = {**META, "group": group, "n_scans_parsed": len(cv["scans"])}
    for k in sorted(cv["scans"]):
        E, i, _ = cv["scans"][k]
        for key, val in cv_scan_peaks(E, i).items():
            prow[f"{key}_s{k}"] = val
        for key, val in cv_scan_background(E, i).items():
            prow[f"{key}_s{k}"] = val
    prow.update(cv_interscan_drift(cv["scans"]))
    frow = {**META, "group": group}
    frow.update(feature_vector(cv))
    return prow, frow


def write_csv(rows, path):
    """Write dict rows with a unified header."""
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(testiso_dir):
    """Process Cv1/Cv2 inside the test_iso folder."""
    peaks, feats = [], []
    for name in ("Cv1", "Cv2"):
        path = Path(testiso_dir) / name / "data.csv"
        if not path.is_file():
            print(f"missing: {path}")
            continue
        prow, frow = rows_for(path, name)
        peaks.append(prow)
        feats.append(frow)
    if not peaks:
        print("no Cv files found")
        return
    write_csv(peaks, "table2_testiso.csv")
    write_csv(feats, "table3_testiso.csv")
    print(f"{len(peaks)} files -> table2_testiso.csv, table3_testiso.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_iso_2")