"""Timed blank-then-sample session: confirm -> wait -> CV -> wait -> CA, with a daily append-mode timeline CSV."""

from __future__ import annotations

import csv
import logging
import time
from datetime import datetime
from pathlib import Path
from serial.tools import list_ports
from tool.transport import find_device, open_device

from korobka.cyclic_voltametry.EmStat4X import EmStat4X
from korobka.connections.base_serial import BaseSerial
from korobka.cyclic_voltametry.exceptions import DeviceNotFoundError

LOG = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
ISO_DIR = ROOT / "test_iso_4"

BAUD_RATE = 921600
TIMEOUT_S = 1.0
MAX_DURATION_S = 480

WITH_TIME = True

WAIT_BEFORE_CV_S = 120
WAIT_BEFORE_CA_S = 180

CV_PARAMS = {
    "e_begin_v": 0, "e_vertex1_v": 0, "e_vertex2_v": 0.6,
    "scan_rate_mv_s": 100, "n_scans": 5, "t_equilibrium_s": 5, "e_step_mv": 5,
}

CA_POTENTIAL_V = 0.4
CA_INTERVAL_S = 0.05
CA_DURATION_S = 120


def probe_ports(baud_rate: int = BAUD_RATE, probe_timeout: float = 0.5) -> None:
    """List every serial port and print its raw response to the 't' handshake."""
    ports = [p.device for p in list_ports.comports()]
    LOG.info("Serial ports found: %s", ports or "none")
    for port in ports:
        if "/dev/tty." in port:
            LOG.info("%s: skipped (tty duplicate)", port)
            continue
        try:
            ser = BaseSerial(port=port, baudrate=baud_rate, parity="N", stopbits=1, timeout=probe_timeout)
        except Exception as exc:
            LOG.info("%s: open failed (%s)", port, exc)
            continue
        try:
            ser.flush_buffers()
            ser.send_binary_command(b"t\n", read_response=False)
            line = ser.readline().decode("ascii", errors="replace")
        finally:
            ser.close()
        LOG.info("%s: response %r -> %s", port, line, "MATCH" if line.startswith("tes4") else "no match")


def _next_index(parent: Path, prefix: str) -> int:
    """Next run number under parent for the given folder-name prefix."""
    if not parent.exists():
        return 1
    nums = [int(p.name[len(prefix):]) for p in parent.iterdir()
            if p.is_dir() and p.name.startswith(prefix) and p.name[len(prefix):].isdigit()]
    return max(nums, default=0) + 1


def _measure(port: str, script_text: str, out_dir: Path) -> tuple[EmStat4X, list[str]]:
    """Write the script into out_dir and run it on a fresh connection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / "script.mscr"
    script_path.write_text(script_text, encoding="ascii")
    em = open_device(port, BAUD_RATE, TIMEOUT_S)
    lines = em.send_methodscript(script_path, max_duration_s=MAX_DURATION_S)
    return em, lines


def run_CV(port: str) -> Path:
    """One CV into the next Cv<n> folder; returns the folder."""
    out_dir = ISO_DIR / f"Cv{_next_index(ISO_DIR, 'Cv')}"
    em, lines = _measure(port, EmStat4X.genmscript_CV(**CV_PARAMS), out_dir)
    em.saveraw_CV(lines, out_dir / "raw.txt")
    dt = CV_PARAMS["e_step_mv"] / CV_PARAMS["scan_rate_mv_s"] if WITH_TIME else None
    em.savecsv_CV(lines, out_dir / "data.csv", dt=dt)
    LOG.info("CV saved to %s", out_dir)
    return out_dir


def run_CA(port: str) -> Path:
    """One CA at CA_POTENTIAL_V into the next Ca<n> folder; returns the folder."""
    out_dir = ISO_DIR / f"Ca{_next_index(ISO_DIR, 'Ca')}"
    em, lines = _measure(port, EmStat4X.genmscript_CA(CA_POTENTIAL_V, CA_INTERVAL_S, CA_DURATION_S), out_dir)
    em.saveraw_CA(lines, out_dir / "raw.txt")
    em.savecsv_CA(lines, out_dir / "data.csv", potential_v=CA_POTENTIAL_V)
    LOG.info("CA saved to %s", out_dir)
    return out_dir


# -------------------------------session flow------------------------------

def _confirm(msg: str) -> None:
    """Block until the operator presses Enter."""
    input(f"\n>>> {msg} -- press Enter to confirm... ")


def _ask_yn(msg: str) -> bool:
    """Ask a y/n question until a valid answer is given."""
    while True:
        ans = input(f"\n>>> {msg} [y/n]: ").strip().lower()
        if ans in ("y", "n"):
            return ans == "y"


def _wait(seconds: int, why: str) -> None:
    """Sleep with a countdown line every 30 s."""
    LOG.info("waiting %d s (%s)", seconds, why)
    remaining = seconds
    while remaining > 0:
        step = min(30, remaining)
        time.sleep(step)
        remaining -= step
        if remaining > 0:
            LOG.info("  %d s left (%s)", remaining, why)


class SessionLog:
    """Append-per-step timeline CSV; reuses today's file across runs, never overwrites."""

    FIELDS = ["step", "solution", "action", "t_start", "t_end", "elapsed_s", "out_dir"]

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                self.step = max(sum(1 for _ in f) - 1, 0)
        else:
            self.step = 0
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def record(self, solution: str, action: str, t_start: datetime,
               t_end: datetime | None = None, out_dir: Path | None = None) -> None:
        """Append one timeline row."""
        t_end = t_end or datetime.now()
        self.step += 1
        row = {"step": self.step, "solution": solution, "action": action,
               "t_start": t_start.isoformat(timespec="seconds"),
               "t_end": t_end.isoformat(timespec="seconds"),
               "elapsed_s": round((t_end - t_start).total_seconds(), 1),
               "out_dir": out_dir.name if out_dir else ""}
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)
        LOG.info("logged: %s %s (%.0f s)", solution, action, row["elapsed_s"])


def _block(port: str, log: SessionLog, solution: str) -> None:
    """load -> wait -> CV -> wait -> CA, all timed."""
    t0 = datetime.now()
    _confirm(f"Load {solution.upper()} onto the electrode")
    log.record(solution, "load", t0)

    t0 = datetime.now()
    _wait(WAIT_BEFORE_CV_S, f"{solution}: settle before CV")
    log.record(solution, f"wait_{WAIT_BEFORE_CV_S}s", t0)

    t0 = datetime.now()
    out = run_CV(port)
    log.record(solution, "CV", t0, out_dir=out)

    t0 = datetime.now()
    _wait(WAIT_BEFORE_CA_S, f"{solution}: relax before CA")
    log.record(solution, f"wait_{WAIT_BEFORE_CA_S}s", t0)

    t0 = datetime.now()
    out = run_CA(port)
    log.record(solution, "CA", t0, out_dir=out)


def run_session(port: str) -> None:
    """Rounds of blank -> rinse -> sample until terminated; closes with rinse + final blank."""
    stamp = datetime.now().strftime("%Y%m%d")
    log = SessionLog(ISO_DIR / f"session_{stamp}.csv")
    LOG.info("run started; log -> %s (continuing at step %d)", log.path, log.step + 1)
    log.record("none", "run_start", datetime.now())

    while True:
        _block(port, log, "blank")

        t0 = datetime.now()
        _confirm("Rinse the electrode with pure water")
        log.record("none", "rinse", t0)

        _block(port, log, "sample")

        if _ask_yn("Terminate the session?"):
            break
        t0 = datetime.now()
        _confirm("Rinse the electrode with pure water (before next round)")
        log.record("none", "rinse", t0)

    t0 = datetime.now()
    _confirm("Rinse the electrode with pure water (closing blank)")
    log.record("none", "rinse", t0)

    _block(port, log, "blank_final")

    log.record("none", "run_end", datetime.now())
    LOG.info("run finished at %s; timeline -> %s",
             datetime.now().isoformat(timespec="seconds"), log.path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        port = find_device()
    except DeviceNotFoundError:
        probe_ports()
        raise
    run_session(port)