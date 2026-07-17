"""Top-level entry: run the per-well screen on an already-initiated dobot.

Bring-up and shutdown are the caller's job, via actions:
    dobot = actions.load_and_confirm()       # construct + initiate + load electrode
    initial_phase(dobot, batch, ...)         # per-batch baseline standard (once)
    run_batch(batch, queue, dobot=dobot)     # the per-well loop
    actions.terminate_and_confirm(dobot)     # hand electrode back + power down
The building blocks it calls live in workflow.py.
"""

from __future__ import annotations

import logging

import Project.config as config
import actions as actions

from notify import Notifier, WorkflowStopped, file_sink
from Project.CV.check import MockChecks
from Project.workflow.workflow1 import measure_well, initial_phase

LOG = logging.getLogger(__name__)


def _disable_in_place(dobot, notifier):
    """Fault path only: power down where it stands -- no move, no claw."""
    try:
        dobot.disable()
    except Exception as exc:
        notifier.notify("error", "disable_failed", error=str(exc))


def run_batch(batch, queue, *, dobot, start_at=None, cv_plate=None, rinse_plate=None,
              pre_rinse=True, check_connection=True, check_clean=True, check_polish=True,
              checks=None, notifier=None):
    """Per-well screen (or resume) on an already-initiated dobot.

    Bring-up (actions.load_and_confirm) and the per-batch initial_phase are the
    caller's job, before this. run_batch is just the per-well loop.

    start_at=None : run the whole queue.
    start_at="A2" : start the queue from that well (manual resume stand-in; the real
                    resume reads the log's completion records, website picks the node).

    Clean finish: home and return, leaving the arm live -- the caller powers down
    (actions.terminate_and_confirm). Fault/freeze: disable in place (no move, no
    claw); re-homing waits for a human-confirmed resume.
    """
    checks = checks or MockChecks()
    notifier = notifier or Notifier(sink=file_sink(config.experiment_dir / batch / "run.log"))

    if start_at is not None:
        names = [it["well"] for it in queue]
        if start_at not in names:
            raise WorkflowStopped("resume_well_not_in_queue", start_at=start_at)
        queue = queue[names.index(start_at):]

    results: dict = {}
    try:
        actions.go_to_generic_safe(dobot)            # position before the run
        for item in queue:
            well = item["well"]
            notifier.notify("info", "well_start", well=well)
            results[well] = measure_well(dobot, well, item["params"], batch=batch,
                                         cv_plate=cv_plate, rinse_plate=rinse_plate,
                                         pre_rinse=pre_rinse, check_connection=check_connection,
                                         check_clean=check_clean, check_polish=check_polish,
                                         checks=checks, notifier=notifier)
            notifier.notify("info", "well_done", well=well)

        notifier.notify("info", "batch_done", batch=batch)
        actions.go_to_generic_safe(dobot)            # clean finish: home; caller powers down
        return results
    except WorkflowStopped as exc:
        notifier.notify("error", "batch_stopped", reason=exc.event, **exc.fields)
        _disable_in_place(dobot, notifier)
        raise
    except Exception as exc:
        notifier.notify("error", "batch_error", error=str(exc))
        _disable_in_place(dobot, notifier)
        raise


# ── manual layer-by-layer test ─────────────────────────────
# Uncomment ONE line in __main__ at a time, simplest first; once it passes, move on.
# Each test brings the arm up (load_and_confirm: construct + initiate + load electrode),
# runs the layer, then powers down (terminate_and_confirm: hand the electrode back).
# Set config.mock = True to dry-run the CV without the potentiostat.

_PARAMS = {"e_begin_v": 0, "e_vertex1_v": -0.3, "e_vertex2_v": 0.3,
           "scan_rate_mv_s": 100, "n_scans": 2, "t_equilibrium_s": 5, "e_step_mv": 5}
_CHECKS = MockChecks(clean_after=1)
_NOTIFIER = Notifier()


def t0_bringup():
    """load_and_confirm -> terminate_and_confirm.  (connect, initiate, home, claw OK?)"""
    d = actions.load_and_confirm()
    actions.terminate_and_confirm(d)


def t1_one_cv():
    """one CV at A1, no rinse, no cleancheck.  (does a measurement run + save?)"""
    d = actions.load_and_confirm()
    measure_well(d, "A1", _PARAMS, batch="t", pre_rinse=False, check_clean=False,
                 checks=_CHECKS, notifier=_NOTIFIER)
    actions.terminate_and_confirm(d)


def t2_rinse_then_cv():
    """add the pre-rinse before the CV."""
    d = actions.load_and_confirm()
    measure_well(d, "A1", _PARAMS, batch="t", check_clean=False,
                 checks=_CHECKS, notifier=_NOTIFIER)
    actions.terminate_and_confirm(d)


def t3_full_well():
    """full single well: rinse -> CV -> cleancheck loop."""
    d = actions.load_and_confirm()
    measure_well(d, "A1", _PARAMS, batch="t", checks=_CHECKS, notifier=_NOTIFIER)
    actions.terminate_and_confirm(d)


def t4_initial_phase():
    """just the per-batch baseline (standard rinse -> wash -> standard CV)."""
    d = actions.load_and_confirm()
    initial_phase(d, "t", _CHECKS, _NOTIFIER)
    actions.terminate_and_confirm(d)


def t5_batch_one():
    """full flow, 1 well: bring up -> initial_phase -> run_batch -> hand back + power down."""
    d = actions.load_and_confirm()
    initial_phase(d, "t", _CHECKS, _NOTIFIER)
    run_batch("t", [{"well": "A1", "params": _PARAMS}], dobot=d,
              checks=_CHECKS, notifier=_NOTIFIER)
    actions.terminate_and_confirm(d)


def t6_batch_two():
    """full flow, 2 wells."""
    d = actions.load_and_confirm()
    initial_phase(d, "t", _CHECKS, _NOTIFIER)
    run_batch("t", [{"well": "A1", "params": _PARAMS},
                    {"well": "A2", "params": _PARAMS}], dobot=d,
              checks=_CHECKS, notifier=_NOTIFIER)
    actions.terminate_and_confirm(d)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    t0_bringup()
    # t1_one_cv()
    # t2_rinse_then_cv()
    # t3_full_well()
    # t4_initial_phase()
    # t5_batch_one()
    # t6_batch_two()