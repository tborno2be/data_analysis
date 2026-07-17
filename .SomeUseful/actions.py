#actions
import time
from logging import getLogger
import Project.config as config
from korobka.robots.Dobot.dobot import Dobot

logger = getLogger(__name__)




def load_and_confirm() -> Dobot:

    dobot = Dobot(ip_address=config.Dobot_ip)
    dobot.clearerror()
    dobot.enable()

    dobot.move(config.generic)   # ensure at generic (returns there if not)
    dobot.set_claw(1, 1)        # open gripper

    logger.info("Gripper open at generic. Waiting for user to load and confirm.")
    input("Load the vial, then press Enter to close the gripper and continue...")

    dobot.set_claw(1, 0)        # close gripper
    logger.info("Gripper closed. Holding at generic, ready to run.")

    return dobot                # return the instance, not the class


def terminate_and_confirm(dobot: Dobot) -> None:

    dobot.move(config.generic)   # already left the vial; just make sure we're at generic

    input("Hold the electrode, then press Enter to open the gripper and terminate...")

    dobot.set_claw(1, 1)        # open gripper (hand the electrode over)
    logger.info("Gripper open at generic. Electrode released.")

    dobot.disable()           # power down
    logger.info("Dobot terminated.")


def go_to_vial(dobot: Dobot, plate, well):
    """Move to 50 mm above a vial."""
    
    
    coords = dobot.get_well_coords(plate, well)
    dobot.speed(config.travel_speed)
    dobot.move(config.generic) 
    dobot.relmove(coords, dz=50)   # move to 50 mm above the well
    dobot.speed(config.approach_speed)
    dobot.relmove(coords)
    logger.info(f"Above vial {well} on plate '{plate}'.")


def leave_vial(dobot: Dobot) -> None:
    """Lift up out of the vial and return to the home position."""

    dobot.speed(config.approach_speed)
    dobot.relmove(dz=50)        # lift up out of the vial

    dobot.speed(config.travel_speed)
    dobot.move(config.generic)         # return home

    logger.info("Left vial; back at generic.")
    
def go_to_wash(dobot: Dobot, wash_dwell_s: float = config.wash_dwell_s) -> None:

    """Dip into the wash station, dwell, then return home."""
    dobot.move(config.generic)  
    dobot.speed(config.travel_speed)
    dobot.relmove(config.wash_station, dz=50)   # 50 mm above the wash station
    dobot.speed(config.approach_speed)
    dobot.relmove(dz=-50)                # descend into the wash station

    logger.info(f"Dwelling in wash station for {config.wash_dwell_s:.1f} s.")
    time.sleep(wash_dwell_s)

    dobot.speed(config.approach_speed)
    dobot.relmove(dz=50)                 # lift back up
    dobot.speed(config.travel_speed)
    dobot.move(config.generic)                  # return home

    logger.info("Wash complete; back at generic.")

def rinse(dobot: Dobot, plate: str, well: str, rinse_dwell_s: float = config.rinse_dwell_s) -> None:
    
    """Rinse a well: descend into the rinse vial, soak, then return home."""
    go_to_vial(dobot, plate, well)   # travel to 50 mm above the rinse well


    logger.info(f"Soaking in rinse well {well} on plate '{plate}' for {config.rinse_dwell_s:.1f} s.")
    time.sleep(rinse_dwell_s)

    leave_vial(dobot)                # lift out and return home


def main() -> None:
    """Connect to the Dobot and run one straight-line pass (no CV yet).

    Mechanical action chain only (measure_CV has moved to the upper-level
    script and is not implemented yet):
        connect -> initiate -> load -> wash -> rinse(A1)
                -> go into vial A1 -> leave -> terminate
    """

    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 1. Connect to dobot (constructing the object opens the TCP connections)
    dobot = Dobot(ip_address=config.Dobot_ip)

    try:
        # 2. Prepare: clear errors, enable, open gripper, go to generic
        dobot.initiate()

        # 3. Load the electrode/vial by hand
        load_and_confirm(dobot)

        # 4. Go to wash station (dip + dwell + back to generic)
        go_to_wash(dobot)

        # 5. Rinse the target well on the rinse rack (dip + dwell + back to generic)
        rinse(dobot, config.default_rinse_plate, "A1")

        # 6. Go down into the target well on the sample plate, then leave
        #    (placeholder for where measure_CV will eventually run)
        go_to_vial(dobot, config.default_cv_plate, "A1")
        leave_vial(dobot)

    finally:
        # Always bring the arm home safely and power down, even on error.
        dobot.terminate()


if __name__ == "__main__":
    #main()
    dobot = Dobot(ip_address=config.Dobot_ip)
    dobot.initiate()
    dobot.move(config.generic)