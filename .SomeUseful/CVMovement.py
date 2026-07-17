"""High-level movement orchestration for automated CV screening on the Dobot.

This module sits *above* the ``Dobot`` class (which already provides the
low-level movement primitives). It exposes four external functions that
compose those primitives into the lab actions needed for the CV workflow:

    - go_to_generic_safe : ensure the arm is at the 'generic' home position
    - go_to_wash         : dip into the wash station, dwell, return home
    - rinse              : dip into the rinse well, dwell, return home
    - go_to_cv           : move to a well, (CV measurement TODO), return home

CV measurement itself is intentionally left empty for now. The measurement
duration is not a fixed sleep -- it will eventually be governed by the
potentiostat signalling that the scan has finished (data stream stops).

Coordinate note
---------------
``Dobot.move()`` already adds ``global_offset`` internally, and
``get_well_coords()`` returns coordinates that the precompute step has
*already* offset. With ``GlobalOffset = [0,0,0,0]`` this is harmless. These
functions deliberately use ``move(coords)`` directly to stay consistent with
the existing ``pickup_vial``/``putdown_vial`` style. If the double-offset is
ever fixed, fix it in one place (precompute layer or move layer), not here.
"""

import time
from logging import getLogger

# BaseTCPIP lives in korobka.connections.base_tcpip, but for connecting we use
# the Dobot subclass (which inherits the TCP/IP layer nd adds movement).
from korobka.robots.Dobot.dobot import Dobot

logger = getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (edit these; coordinates are NOT stored here -- they are
# looked up at runtime from the Dobot's precomputed absolute_wells)
# ---------------------------------------------------------------------------

Dobot_ip = "192.168.1.6"

cv_plate = "off_deck_8"     # plate holding the wells under test + standard
rinse_plate = "rinse_rack"  # TODO: replace with the real rinse-plate name in positions.json

wash_station = "washstation"  # named position in positions.json
generic = "genericCV"           # home / neutral named position

wash_dwell_s = 5.0   # seconds to dwell in the wash station
rinse_dwell_s = 5.0  # seconds to dwell in the rinse well

travel_speed = 10    # speed for moving to generic 
approach_speed = 10  # speed for moving to specific position



# ---------------------------------------------------------------------------
# Movement actions (each one: ensure home -> do the thing -> return home)
# ---------------------------------------------------------------------------

def go_to_generic_safe(dobot: Dobot) -> None:
    """Ensure the arm is at the 'generic' home position.

    If it is already within precision of 'generic', do nothing. Otherwise
    move it back to 'generic'. Used as the opening guard for every action so
    each one starts from a known, clean position.
    """
    if dobot.amithere(generic):
        logger.debug(f"Already at generic)
        return
    logger.info("Not at '%s' -- returning home.", generic)
    dobot.speed(travel_speed)
    dobot.move(generic)

def load_and_confirm(dobot: Dobot) -> None:
    """Manual loading step: ensure home, open gripper, wait for the user to
    confirm, then close the gripper and stay at generic, ready to run.

    Called as the first action after the Dobot is connected and initiated
    (so the arm is already enabled). Blocks on user input -- during the wait
    the arm is enabled, gripper open, hovering at generic.
    """
    go_to_generic_safe(dobot)   # ensure at generic (returns there if not)

    dobot.set_claw(1, 1)        # open gripper
    logger.info("Gripper open at generic. Waiting for user to load and confirm.")

    input("Load the vial, then press Enter to close the gripper and continue...")

    dobot.set_claw(1, 0)        # close gripper
    logger.info("Gripper closed. Holding at generic, ready to run.")


def go_to_wash(dobot: Dobot, dwell_s: float = wash_dwell_s) -> None:
    """Dip into the wash station, dwell, then return to generic.

    :param dwell_s: How long to stay in the wash station (seconds).
    """
    go_to_generic_safe(dobot)

    dobot.speed(travel_speed)
    dobot.relmove(wash_station, dz= 50)
    dobot.speed(approach_speed)
    dobot.relmove(dz = -50)   # descend to the wash-station coordinate (no extra plunge yet)

    logger.info("Dwelling in wash station for %.1f s", dwell_s)
    time.sleep(dwell_s)

    dobot.relmove(dz=50)
    dobot.speed(travel_speed)
    dobot.move(generic)        # return home
    logger.info("Wash complete; back at generic.")


def rinse(dobot: Dobot, well: str, dwell_s: float = rinse_dwell_s) -> None:
    """Rinse: go to the same-named well on the rinse plate, dwell, return home.

    The rinse well shares the target well's name but lives on RINSE_PLATE,
    so rinsing well 'A1' goes to RINSE_PLATE's 'A1'.

    :param well: Well ID (e.g. 'A1'), same name as the well being tested.
    :param dwell_s: How long to soak in the rinse well (seconds).
    """
    go_to_generic_safe(dobot)

    coords = dobot.get_well_coords(rinse_plate, well)
    dobot.speed(travel_speed)
    dobot.relmove(coords, dz= 50)
    dobot.speed(approach_speed)
    dobot.relmove(dz=-50)         # descend to the rinse-well coordinate (no extra plunge yet)

    logger.info("Soaking in rinse well %s for %.1f s", well, dwell_s)
    time.sleep(dwell_s)

    dobot.relmove(dz=50)
    dobot.speed(travel_speed)
    dobot.move(generic)        # return home
    logger.info("Rinse of %s complete; back at generic.", well)


def go_to_cv(dobot: Dobot, well: str) -> None:
    """Move to a well on the CV plate, run the CV measurement, then return home.

    Used for both the real measurement wells and the standard (B4).

    CV measurement is intentionally empty for now. The dwell here is NOT a
    fixed sleep -- when wired to the potentiostat, this is where we will start
    the scan and wait for it to finish (data stream stops). Peak / no-peak
    judgement happens *after* the scan, elsewhere.

    :param well: Well ID on CV_PLATE (e.g. 'A1' to test, 'B4' for the standard).
    """
    go_to_generic_safe(dobot)

    coords = dobot.get_well_coords(cv_plate, well)

    dobot.speed(travel_speed)
    dobot.relmove(coords, dz= 50)
    dobot.speed(approach_speed)
    dobot.relmove(dz=-50)            # descend to the well coordinate (no extra plunge yet)

    # -----------------------------------------------------------------
    # TODO: trigger CV measurement here and wait for it to finish.
    #       Duration is decided by the potentiostat (scan end / data stop),
    #       NOT by a fixed sleep. Use the existing send_n_wait mechanism.
    # -----------------------------------------------------------------
    logger.info("At well %s on %s -- CV measurement placeholder (not implemented).", well, cv_plate)
    time.sleep(5)

    dobot.relmove(dz=50)
    dobot.speed(travel_speed)
    dobot.move(generic)        # return home
    logger.info("Done at well %s; back at generic.", well)


# ---------------------------------------------------------------------------
# Connect + single straight-line run (wash -> rinse one well -> test one well)
# ---------------------------------------------------------------------------


def main() -> None:
    """Connect to the Dobot and run one straight-line pass.

    This version runs a single linear sequence (no standard / cleaning loop):
        connect -> initiate -> wash -> rinse(A1) -> test CV(A1) -> terminate
    """
    # 1. Connect to dobot (constructing the object opens the TCP connections)
    dobot = Dobot(ip_address=Dobot_ip)

    try:
        # 2. Prepare: clear errors, enable, open gripper, go to generic
        dobot.initiate(home="genericCV")

        # 3. Load the gripper by hand
        load_and_confirm(dobot)

        # 4. Go to wash station (dip + dwell + back to generic)
        go_to_wash(dobot)

        # 5. Rinse the target well (dip + dwell + back to generic)
        rinse(dobot, "A1")

        # 6. Go to the target well and test CV (measurement empty for now)
        go_to_cv(dobot, "A1")

    finally:
        # Always bring the arm home safely and power down, even on error.
        dobot.terminate(home="genericCV")


if __name__ == "__main__":
    main()