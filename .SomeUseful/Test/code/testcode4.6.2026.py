"""
Hardware bring-up / functional test for the EmStat4X CV driver (no-Instrument version).

Place this file at:    CVLABTEST/Test/code/test_cv.py
Outputs are written to: CVLABTEST/Test/Result/   ->  test1_raw.txt, test1.csv, test1_method.mscr

It locates the project automatically from its own location, so run it straight
from the IDE or with:   python test_cv.py

Per-run editing happens in the CONFIG block:
  - LABEL : output base name. Change test1 -> test2 / test3 / ... each run.
  - MOCK  : True  = dry run, no hardware (checks imports / paths / file writing).
            False = real measurement on the connected EmStat4X.

Tip: run once with MOCK = True to confirm the plumbing, then flip to False and
plug in the instrument.
"""

import os
import sys
import logging
import traceback

# ──────────────────────────────────────────────────────────
# CONFIG  --  edit these per run
# ──────────────────────────────────────────────────────────
LABEL = "test1"            # -> test1_raw.txt, test1.csv, test1_method.mscr
MOCK  = False              # True = dry run (no hardware); False = real measurement

PORT     = "COM4"          # serial port of the EmStat4X
BAUDRATE = 921600
TIMEOUT  = 1               # SHORT read timeout (s), paired with the readline loop

# Hard cap on total measurement wait (s). Must be LARGER than the longest a
# single run can take, or a normal measurement gets cut short. Rough estimate:
#   t_equilibrium + (total potential travel / scan_rate) * n_scans
# e.g. begin 0, v1 -0.2, v2 0.6 -> travel 1.6 V; at 0.05 V/s -> ~32 s/scan.
# When in doubt set it generously large; it only exists to escape a stall.
MAX_DURATION_S = 480

RUN_PORT_CHECK = True      # before the run, open+close the port once to confirm
                           # it exists / is free / framing is accepted. Set False
                           # if the re-open right before the run causes trouble.

# CV parameters
E_BEGIN_V       =  0.0
E_VERTEX1_V     = -0.2
E_VERTEX2_V     =  0.6
SCAN_RATE_MV_S  =  50
N_SCANS         =  1
T_EQUILIBRIUM_S =  10
E_STEP_MV       =  5

# ──────────────────────────────────────────────────────────
# Path setup  --  make the project's modules importable
# ──────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))           # .../Test/code
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))      # .../CVLABTEST
CV_DIR       = os.path.join(PROJECT_ROOT, "cyclic voltammetry")     # folder name has a space
RESULT_DIR   = os.path.abspath(os.path.join(HERE, "..", "Result"))  # .../Test/Result

# PROJECT_ROOT -> resolves `korobka.connections.base_serial` and `palmsens.mscript`
# CV_DIR       -> resolves `CVMeasurement` (imported as a bare module)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, CV_DIR)

from Project.CVMeasurement import EmStat4X                       # noqa: E402

# ──────────────────────────────────────────────────────────
# Logging  --  so the driver's LOG.info() is visible
# ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_cv")


def banner(text: str) -> None:
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def main() -> None:
    os.makedirs(RESULT_DIR, exist_ok=True)
    raw_path    = os.path.join(RESULT_DIR, f"{LABEL}_raw.txt")
    csv_path    = os.path.join(RESULT_DIR, f"{LABEL}.csv")
    script_path = os.path.join(RESULT_DIR, f"{LABEL}_method.mscr")

    banner(f"CONFIG   LABEL={LABEL}   MOCK={MOCK}")
    print(f"  PORT={PORT}  BAUDRATE={BAUDRATE}  TIMEOUT={TIMEOUT}s  MAX_DURATION={MAX_DURATION_S}s")
    print(f"  CV: begin={E_BEGIN_V} V  v1={E_VERTEX1_V} V  v2={E_VERTEX2_V} V  "
          f"rate={SCAN_RATE_MV_S} mV/s  scans={N_SCANS}  t_eq={T_EQUILIBRIUM_S}s  step={E_STEP_MV} mV")
    print(f"  Results -> {RESULT_DIR}")

    # ── 1) EmStat4X.__init__ ──────────────────────────────
    banner("[1] EmStat4X.__init__")
    dev = EmStat4X(device_port=PORT, baud_rate=BAUDRATE, timeout=TIMEOUT, mock=MOCK)
    print(f"  OK   port={dev.port}   baudrate={dev.baudrate}   mock={dev.mock}")

    # ── 2) generate_methodscript  (pure, no hardware) ─────
    banner("[2] generate_methodscript")
    script_text = EmStat4X.generate_methodscript(
        e_begin_v       = E_BEGIN_V,
        e_vertex1_v     = E_VERTEX1_V,
        e_vertex2_v     = E_VERTEX2_V,
        scan_rate_mv_s  = SCAN_RATE_MV_S,
        n_scans         = N_SCANS,
        t_equilibrium_s = T_EQUILIBRIUM_S,
        e_step_mv       = E_STEP_MV,
    )
    with open(script_path, "w", encoding="ascii", newline="\n") as f:
        f.write(script_text)
    print(f"  OK   MethodSCRIPT -> {script_path}")
    print("  --- preview ---")
    for line in script_text.splitlines():
        print("    " + line)

    # ── 3) Port-open check  (hardware only) ───────────────
    # Open the port once with the real framing (8-N-1) and close it, to confirm
    # the COM port exists, is free, and the settings are accepted -- fails fast
    # with a clear error before committing to a full scan.
    if MOCK:
        banner("[3] Port-open check  --  SKIPPED (MOCK)")
    elif not RUN_PORT_CHECK:
        banner("[3] Port-open check  --  SKIPPED (disabled in CONFIG)")
    else:
        banner("[3] Port-open check  (open + close)")
        import serial                                              # noqa: E402
        from korobka.connections.base_serial import BaseSerial     # noqa: E402
        ser = BaseSerial(
            port=PORT,
            baudrate=BAUDRATE,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT,
            rtscts=False,
        )
        print(f"  is_open = {ser._ser.is_open}")
        ser.close()
        print("  OK   port opened and closed cleanly")

    # ── 4) send_methodscript  (the real measurement) ──────
    banner("[4] send_methodscript  (running CV)")
    result_lines = dev.send_methodscript(script_path, max_duration_s=MAX_DURATION_S)
    n = len(result_lines)
    print(f"  received {n} result line(s)")
    if not MOCK and n < 5:
        print("  WARNING: very few lines -- likely an error / wrong port / no data. Check the raw file below.")

    # ── 5) save_raw_result ────────────────────────────────
    banner("[5] save_raw_result")
    dev.save_raw_result(result_lines, raw_path)
    print(f"  OK   -> {raw_path}")

    # ── 6) save_csv_result ────────────────────────────────
    banner("[6] save_csv_result")
    dev.save_csv_result(result_lines, csv_path)
    print(f"  OK   -> {csv_path}")

    # ── 7) log_cv_result  (verbose: logs every package) ───
    banner("[7] log_cv_result")
    curves = dev.log_cv_result(result_lines)
    if curves is None:
        print("  (mock mode: parsing skipped)")
    else:
        n_loops  = len(curves.loops)
        n_points = sum(len(loop.packages) for loop in curves.loops)
        print(f"  OK   loops={n_loops}   total points={n_points}")

    banner("DONE")
    print(f"  raw : {raw_path}")
    print(f"  csv : {csv_path}")
    print(f"  mscr: {script_path}")
    print("  -> open the raw file to confirm the run looks complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        banner("FAILED")
        traceback.print_exc()
        sys.exit(1)