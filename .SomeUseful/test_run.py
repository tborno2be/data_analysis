#python -m Project.workflow.test_run

import Project.config as cfg
import Project.workflow.workflow1 as wf

EXP = "test_workflow_t6_3"
RACKS = cfg.BATCHES["batch1"]
PARAMS = {
    "e_begin_v": 0, "e_vertex1_v": -0.3, "e_vertex2_v": 0.3,
    "scan_rate_mv_s": 100, "n_scans": 2, "t_equilibrium_s": 5, "e_step_mv": 5,
}
CHECKS = wf.MockChecks()
NOTIFIER = wf.Notifier()
ROOT = wf.experiment_root(EXP)


def t0_bringup():
    d = wf.load_and_confirm()
    wf.terminate_and_confirm(d)


def t1_move():
    d = wf.load_and_confirm()
    wf.go_to_vial(d, RACKS["sample"], "A1")
    wf.leave_vial(d)
    wf.terminate_and_confirm(d)


def t2_wash_rinse():
    d = wf.load_and_confirm()
    wf.go_to_wash(d)
    wf.rinse(d, RACKS["rinse"], "A1")
    wf.terminate_and_confirm(d)


def t3_measure_cv():
    d = wf.load_and_confirm()
    wf.measure_cv(d, RACKS["sample"], "A1", PARAMS, ROOT / "A1_cv", CHECKS, NOTIFIER)
    wf.terminate_and_confirm(d)

def t3_measure_cv2():
    d = wf.load_and_confirm()
    res = wf.measure_cv(d, RACKS["sample"], "A1", PARAMS, ROOT / "A1_cv", CHECKS, NOTIFIER)
    for scan, pk in res.results.items():
        print(scan, "clean" if pk is None or pk.is_clean else f"Epa={pk.e_pc} E_half={pk.e_half}") 
    wf.terminate_and_confirm(d)

def t4_initial_phase():
    d = wf.load_and_confirm()
    wf.initial_phase(d, "batch1", ROOT, CHECKS, NOTIFIER)
    wf.terminate_and_confirm(d)


def t5_measure_well():
    d = wf.load_and_confirm()
    wf.measure_well(d, "A1", PARAMS, batch="batch1", out_root=ROOT, checks=CHECKS, notifier=NOTIFIER)
    wf.terminate_and_confirm(d)


def t6_run():
    wf.run(EXP, "batch1", [{"well": "A1", "params": PARAMS}])


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    #t0_bringup()
    #t1_move()
    #t2_wash_rinse()
    #t3_measure_cv2() 
    #t4_initial_phase()
    #t5_measure_well()
    t6_run()

    #3456都会有新文件夹回头
#     t3 重跑、t4 重跑 → 覆盖。
# t5 重跑、t6 重跑 → 不覆盖(分层)。
# t4 和 t6 互相之间 → 会覆盖对方的 baseline_standard/。

