"""Run all extractions: CA (table 1), CV peaks+features (tables 2-3), and the unified diagnostic (tables 4-5)."""

import logging
from pathlib import Path

from analysis import build_ca_table, build_cv_tables, build_diagnostic_tables, export_pointwise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root = Path(__file__).parent
    build_ca_table(root / "CA", root / "out" / "table1_ca.csv")
    build_cv_tables(root / "CV", root / "out" / "table2_cv_peaks.csv", root / "out" / "table3_cv_feats.csv")
    build_diagnostic_tables(root / "CV", root / "out")
    export_pointwise(root / "CV", root / "out")