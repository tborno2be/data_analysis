from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import shutil

import Project.config as cfg
import Project.electrochem.peak as peak
import Project.electrochem.storage as store
from Project.electrochem.connection import MockChecks
from Project.workflow.notify import Notifier, WorkflowStopped, file_sink
from korobka.cyclic_voltametry.EmStat4X import EmStat4X
from korobka.robots.Dobot.dobot import Dobot

LOG = logging.getLogger(__name__)




#_______________Basic dobot movement_________________


def load_and_confirm() -> Dobot:
    """Construct + power on the Dobot, open gripper for the user to load the electrode, close, return it."""
    dobot = Dobot(ip_address=cfg.Dobot_ip)
    dobot.clearerror()
    dobot.enable()

    dobot.move(cfg.generic)
    dobot.set_claw(1, 1)        # open gripper
    LOG.info("Gripper open at generic. Waiting for user to load and confirm.")
    input("Load the vial, then press Enter to close the gripper and continue...")

    dobot.set_claw(1, 0)        # close gripper
    LOG.info("Gripper closed. Holding at generic, ready to run.")
    return dobot


def terminate_and_confirm(dobot: Dobot) -> None:
    """At generic, wait for the user to take the electrode, open gripper, power down."""
    dobot.move(cfg.generic)
    input("Hold the electrode, then press Enter to open the gripper and terminate...")

    dobot.set_claw(1, 1)        # open gripper (hand the electrode over)
    LOG.info("Gripper open at generic. Electrode released.")
    dobot.disable()
    LOG.info("Dobot terminated.")


def go_to_vial(dobot: Dobot, plate, well) -> None:
    """Travel to the well and descend into it."""
    coords = dobot.get_well_coords(plate, well)
    dobot.speed(cfg.travel_speed)
    dobot.move(cfg.generic)
    dobot.relmove(coords, dz=50)        # 50 mm above the well
    dobot.speed(cfg.approach_speed)
    dobot.relmove(coords)               # descend into the well
    LOG.info(f"At vial {well} on plate '{plate}'.")


def leave_vial(dobot: Dobot) -> None:
    """Lift out of the vial and return to generic."""
    dobot.speed(cfg.approach_speed)
    dobot.relmove(dz=50)                # lift out
    dobot.speed(cfg.travel_speed)
    dobot.move(cfg.generic)             # return home
    LOG.info("Left vial; back at generic.")


def go_to_wash(dobot: Dobot, wash_dwell_s: float = cfg.wash_dwell_s) -> None:
    """Dip into the wash station, dwell, then return to generic."""
    dobot.move(cfg.generic)
    dobot.speed(cfg.travel_speed)
    dobot.relmove(cfg.wash_station, dz=50)   # 50 mm above the wash station
    dobot.speed(cfg.approach_speed)
    dobot.relmove(dz=-50)                     # descend in

    LOG.info(f"Dwelling in wash station for {wash_dwell_s:.1f} s.")
    time.sleep(wash_dwell_s)

    dobot.speed(cfg.approach_speed)
    dobot.relmove(dz=50)                # lift out
    dobot.speed(cfg.travel_speed)
    dobot.move(cfg.generic)            # return home
    LOG.info("Wash complete; back at generic.")


def rinse(dobot: Dobot, plate: str, well: str, rinse_dwell_s: float = cfg.rinse_dwell_s) -> None:
    """Descend into a rinse well, soak, then return to generic."""
    go_to_vial(dobot, plate, well)
    LOG.info(f"Soaking in rinse well {well} on plate '{plate}' for {rinse_dwell_s:.1f} s.")
    time.sleep(rinse_dwell_s)
    leave_vial(dobot)


def _disable_in_place(dobot, notifier):
    """Fault path only: power down where it stands -- no move, no claw."""
    try:
        dobot.disable()
    except Exception as exc:
        notifier.notify("error", "disable_failed", error=str(exc))


#_______________analysis data and file storage_________________


@dataclass
class MeasureResult:
    well: str
    plate: str
    out_dir: Path
    results: dict[int, peak.PeakResult | None]   # per scan; website overlays into one plot


def _read_scans(csv_path: Path) -> dict[int, tuple[list, list, list]]:
    """data.csv -> {scan: (E, i, t)}; equilibration rows (blank scan) skipped."""
    scans: dict = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["scan"]:
                continue
            E, i, t = scans.setdefault(int(row["scan"]), ([], [], []))
            E.append(float(row["potential"]))
            i.append(float(row["current"]))
            t.append(float(row["time"]))
    return scans


def _analyze(csv_path: Path, e_step_mv: float, scan_rate_mv_s: float) -> dict[int, peak.PeakResult | None]:
    """find_peak on every scan present -> {scan: PeakResult or None}."""
    out: dict = {}
    try:
        scans = _read_scans(csv_path)
    except Exception as exc:
        LOG.warning(f"read scans failed: {exc}")
        return out
    for s in sorted(scans):
        E, i, t = scans[s]
        if len(E) < 10:
            out[s] = None
            continue
        try:
            out[s] = peak.find_peak(E, i, t=t, scan_rate=scan_rate_mv_s, e_step=e_step_mv)
        except Exception as exc:
            LOG.warning(f"find_peak failed on scan {s}: {exc}")
            out[s] = None
    return out

def experiment_root(experiment: str, batch: str | None = None) -> Path:
    """Top builds the output base with this: experiment[/batch]. Lower layers just receive it."""
    root = cfg.experiment_dir / experiment
    return root / batch if batch else root


def resolve_run_dir(well_dir: Path) -> Path:
    """Dir to write this measurement into; lazily split the flat 1st run into run001 on the 2nd."""
    well_dir = Path(well_dir)
    runs = [p for p in well_dir.glob(cfg.run_prefix + "[0-9]*") if p.is_dir()]
    if runs:                                       # already split -> next run
        nxt = max(int(p.name[len(cfg.run_prefix):]) for p in runs) + 1
        run = well_dir / cfg.run_fmt.format(nxt)
        run.mkdir(parents=True)
        return run
    if (well_dir / cfg.data_marker).exists():      # flat 1st run present -> split
        run1 = well_dir / cfg.run_fmt.format(1)
        run1.mkdir()
        for child in list(well_dir.iterdir()):
            if child != run1:
                shutil.move(str(child), str(run1 / child.name))
        run2 = well_dir / cfg.run_fmt.format(2)
        run2.mkdir()
        return run2
    well_dir.mkdir(parents=True, exist_ok=True)    # first ever / empty
    return well_dir


#_______________basic measure function_________________


def measure_cv(dobot, plate, well, params, out_dir, checks, notifier, *, check_connection=True) -> MeasureResult:
    """Position -> run CV -> leave -> save raw/csv -> find_peak."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    e_step_mv = params.get("e_step_mv", 5)
    scan_rate_mv_s = params["scan_rate_mv_s"]

    script_path = out_dir / "script.mscr"
    script_path.write_text(EmStat4X.generate_methodscript(**params), encoding="ascii")

    go_to_vial(dobot, plate, well)

    # connection is checked here, AFTER the electrode is in place
    if check_connection and not checks.is_connected():
        notifier.notify("error", "not_connected", plate=plate, well=well)
        raise WorkflowStopped("not_connected", plate=plate, well=well)

    # on any failure we do NOT leave_vial: freeze in place, let the top level disable()
    if cfg.mock:
        lines = ["mock\n"]
    else:
        emstat = EmStat4X(cfg.device_port, cfg.baud_rate, cfg.timeout)
        lines = emstat.send_methodscript(script_path, max_duration_s=cfg.max_duration_s)
    leave_vial(dobot)

    store.save_raw(lines, out_dir / "raw.txt")
    csv_path = store.save_csv(lines, out_dir / "data.csv",
                              e_step_mv=e_step_mv, scan_rate_mv_s=scan_rate_mv_s)
    results = _analyze(csv_path, e_step_mv, scan_rate_mv_s)
    return MeasureResult(well, plate, out_dir, results)


def cleancheck(dobot, well, run_dir, batch, checks, notifier, *,
               check_connection=True, check_polish=True) -> MeasureResult:
    """wash -> rinse(well) -> standard CV -> is_clean? looped until clean or limit."""
    racks = cfg.BATCHES[batch]
    for attempt in range(1, cfg.clean_max_retries + 1):
        go_to_wash(dobot)
        rinse(dobot, racks["rinse"], well)
        std = measure_cv(dobot, racks["sample"], cfg.standard_well_cv, cfg.standard_params,
                         run_dir / cfg.cleancheck_fmt.format(attempt), checks, notifier,
                         check_connection=check_connection)
        if check_polish and checks.needs_polish(std.results):
            notifier.notify("error", "polish_required", well=well, attempt=attempt)
            terminate_and_confirm(dobot)
            raise WorkflowStopped("polish_required", well=well, attempt=attempt)
        if checks.is_clean(std.results, well=well):
            notifier.notify("info", "cleancheck_passed", well=well, attempt=attempt)
            return std
        notifier.notify("warning", "cleancheck_retry", well=well, attempt=attempt)
    notifier.notify("error", "cleancheck_failed", well=well, attempts=cfg.clean_max_retries)
    raise WorkflowStopped("cleancheck_failed", well=well, attempts=cfg.clean_max_retries)




#_______________measure function of a complete well_________________

def measure_well(
    dobot,
    well,
    params,
    *,
    batch,
    out_root,
    pre_rinse=True,
    check_connection=True,
    check_clean=True,
    check_polish=True,
    checks=None,
    notifier=None,
):
    """One well: [pre-rinse] -> sample CV -> [cleancheck]; lazily run-split on re-measure."""

    if checks is None:
        checks = MockChecks()
    if notifier is None:
        notifier = Notifier()

    racks = cfg.BATCHES[batch]

    run_dir = resolve_run_dir(Path(out_root) / well)

    checks.reset_well(well)
    if pre_rinse:
        rinse(dobot, racks["rinse"], well)

    sample = measure_cv(
        dobot, racks["sample"], well, params, run_dir, checks, notifier,
        check_connection=check_connection,
    )

    std = None
    if check_clean:
        std = cleancheck(
            dobot, well, run_dir, batch, checks, notifier,
            check_connection=check_connection,
            check_polish=check_polish,
        )
    return {"sample": sample, "cleancheck": std}

#_______________initial step of cv test: wash-rinse-standard_________________


def initial_phase(
    dobot,
    batch,
    out_root,
    checks,
    notifier,
    *,
    check_connection=True,
):
    """Once per batch: standard rinse -> wash -> baseline standard CV."""
    racks = cfg.BATCHES[batch]
    go_to_wash(dobot)
    rinse(dobot, racks["rinse"], cfg.standard_well_rinse)

    return measure_cv(
        dobot,
        racks["sample"],
        cfg.standard_well_cv,
        cfg.standard_params,
        Path(out_root) / "baseline_standard",
        checks,
        notifier,
        check_connection=check_connection,
    )


#_______________the top function_________________

def run(experiment, batch, queue, *, checks=None, notifier=None):
    """Parse inputs -> bring up -> initial_phase -> measure each well -> bring down.
    Clean finish: terminate (hand electrode back). Fault: disable in place, then raise."""

    out_root = experiment_root(experiment)          # top decides: no batch subfolder here
    
    if checks is None:
        checks = MockChecks()
    if notifier is None:
        notifier = Notifier()

    dobot = load_and_confirm()
    try:
        initial_phase(dobot, batch, out_root, checks, notifier)
        results = {}
        for item in queue:
            results[item["well"]] = measure_well(
                dobot, item["well"], item["params"],
                batch=batch, out_root=out_root,
                checks=checks, notifier=notifier,
            )
    except Exception:
            _disable_in_place(dobot, notifier)
            raise

    rinse(dobot, cfg.BATCHES[batch]["rinse"], cfg.standard_well_rinse)   # final rinse at standard
    terminate_and_confirm(dobot)
    return results


#_______________test_________________

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    _params = {
        "e_begin_v": 0, "e_vertex1_v": -0.3, "e_vertex2_v": 0.3,
        "scan_rate_mv_s": 100, "n_scans": 2, "t_equilibrium_s": 5, "e_step_mv": 5,
    }
    
    queue = [
        {"well": "A1", "params": _params},
    ]

    results = run("exp0617", "batch1", queue)
    print(results)