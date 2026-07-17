"""Batch device test: five CV runs into test/CV and five CA runs into test/CA."""

from __future__ import annotations

import logging
from pathlib import Path

from korobka.cyclic_voltametry.EmStat4X import EmStat4X
from test_2.transport import find_device, open_device

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
CV_DIR = ROOT / "CV"
CA_DIR = ROOT / "CA"

BAUD_RATE = 921600
TIMEOUT_S = 1.0
MAX_DURATION_S = 480

CV_PARAMS = {
    "e_begin_v": 0, "e_vertex1_v": -0.3, "e_vertex2_v": 0.3,
    "scan_rate_mv_s": 100, "n_scans": 3, "t_equilibrium_s": 5, "e_step_mv": 5,
}
CV_REPEATS = 5

CA_POTENTIALS_V = [0.4, 0.4, 0.4, 0.4, 0.4]
CA_INTERVAL_S = 0.05
CA_DURATION_S = 10


def _next_index(parent: Path, prefix: str = "") -> int:
    """Next run number under parent for the given folder-name prefix."""
    if not parent.exists():
        return 1
    nums = [int(p.name[len(prefix):]) for p in parent.iterdir()
            if p.is_dir() and p.name.startswith(prefix) and p.name[len(prefix):].isdigit()]
    return max(nums, default=0) + 1


# def _measure(port: str, script_text: str, out_dir: Path) -> tuple[EmStat4X, list[str]]:
#     """Write the script into out_dir and run it on a fresh connection."""
#     out_dir.mkdir(parents=True, exist_ok=True)
#     script_path = out_dir / "script.mscr"
#     script_path.write_text(script_text, encoding="ascii")
#     em = EmStat4X(port, BAUD_RATE, TIMEOUT_S)
#     lines = em.send_methodscript(script_path, max_duration_s=MAX_DURATION_S)
#     return em, lines

def _measure(port: str, script_text: str, out_dir: Path) -> tuple[EmStat4X, list[str]]:
    """Write the script into out_dir and run it on a fresh connection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / "script.mscr"
    script_path.write_text(script_text, encoding="ascii")
    em = open_device(port, BAUD_RATE, TIMEOUT_S)
    lines = em.send_methodscript(script_path, max_duration_s=MAX_DURATION_S)
    return em, lines

def run_CV(port: str) -> None:
    """Five identical CV runs; even-numbered folders get the time column."""
    dt = CV_PARAMS["e_step_mv"] / CV_PARAMS["scan_rate_mv_s"]
    for _ in range(CV_REPEATS):
        n = _next_index(CV_DIR)
        out_dir = CV_DIR / str(n)
        em, lines = _measure(port, EmStat4X.genmscript_CV(**CV_PARAMS), out_dir)
        em.saveraw_CV(lines, out_dir / "raw.txt")
        em.savecsv_CV(lines, out_dir / "data.csv", dt=dt if n % 2 == 0 else None)
        LOG.info("CV run %d saved (%s time column)", n, "with" if n % 2 == 0 else "without")


def run_CA(port: str) -> None:
    """Five CA runs at fixed different potentials."""
    for potential_v in CA_POTENTIALS_V:
        n = _next_index(CA_DIR)
        out_dir = CA_DIR / str(n)
        em, lines = _measure(port, EmStat4X.genmscript_CA(potential_v, CA_INTERVAL_S, CA_DURATION_S), out_dir)
        em.saveraw_CA(lines, out_dir / "raw.txt")
        em.savecsv_CA(lines, out_dir / "data.csv", potential_v=potential_v)
        LOG.info("CA run %d saved at %.3f V", n, potential_v)


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     port = EmStat4X.find_port()
#     run_CV(port)
#     run_CA(port)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = find_device()
    run_CV(port)
    run_CA(port)