#!/usr/bin/env python3
"""
CV 定时测量 —— 只跑一个 CV 轮次，只调用 EmStat4X。

做什么：
  从开始起反复跑循环伏安(CV)，每测完一次等固定秒数(GAP_S，默认 30s)再测下一次，
  持续一个轮次时长(ROUND_DURATION_MIN，默认 10 分钟)。满时长后不再开新测量，
  正在跑的那次跑完为止。

时间戳日志（共享）：
  本脚本和 ca_measure.py 当天写进【同一个】 analysis_<日期>/session.csv。
  第一个运行的脚本创建它，之后的（无论 CV 还是 CA）都【追加】进去，
  cv_XXXX / ca_XXXX 的编号自动接着已有的往下排。这样 CV 和 CA 的时间戳在一个文件里。

文件放哪儿：
  本脚本和 ca_measure.py 都放在 data_analysis 根目录（和 analysis_<日期>、tool 平级），
  不要放进 tool/ 里。两个脚本必须在同一目录，才能共享同一个 session。

连接设备用 data_analysis 下 tool/transport.py 的 find_device()/open_device()，底层是
korobka.cyclic_voltametry.EmStat4X。

输出（都在 analysis_<日期>/ 下）：
  cv_0001/, cv_0002/ ...  每次 CV：data.csv(scan,potential,current,time) + raw.txt
  session.csv             共享时间戳日志

怎么跑：
  cd 到 data_analysis，然后  python cv_timed_measure.py
  想不接仪器只验证流程/计时/文件结构，把 SIMULATE 改成 True。
"""

from __future__ import annotations

# =====================================================================
#  参数 —— 在这里填（默认 0-0.5V，3 圈，100 mV/s）
# =====================================================================

E_BEGIN_V        = 0.0    # 起始电位 (V)
E_VERTEX1_V      = 0.5    # 顶点 1 (V)   —— 与 E_BEGIN 一起决定 0–0.5 扫窗
E_VERTEX2_V      = 0.0    # 顶点 2 (V)   —— 扫回的下顶点
N_SCANS          = 3      # 扫描圈数
SCAN_RATE_MV_S   = 100    # 扫速 (mV/s)
E_STEP_MV        = 5      # 电位步长 (mV)；时间轴 dt = E_STEP / SCAN_RATE = 0.05 s
T_EQUILIBRIUM_S  = 2      # 每次 CV 前在起始电位的平衡保持时长 (s)

ROUND_DURATION_MIN = 10   # 一个 CV 轮次多长（分钟）
GAP_S              = 30   # 每次“测完”之后等待多少秒，再开始下一次

MAX_MEAS_DURATION_S = 300 # 单次测量最长等待(秒)；设备发回结束行就立刻返回，只是防卡死上限

OUTPUT_PARENT   = None    # None = 自动用脚本所在目录(data_analysis)；也可写死绝对路径
ANALYSIS_PREFIX = "analysis"   # 共享文件夹名前缀：<prefix>_<MMDDYYYY>

SIMULATE          = False # True = 不接仪器，用合成数据+短延时验证流程
SIM_MEAS_SECONDS  = 1.0   # 干跑时每次“测量”假装花多少秒

# =====================================================================
#  以下一般不用改
# =====================================================================

import csv
import sys
import time
import math
import logging
import importlib.util
from datetime import datetime
from pathlib import Path

LOG = logging.getLogger("cv_timed_measure")
HERE = Path(__file__).resolve().parent
DT_S = E_STEP_MV / SCAN_RATE_MV_S


def _ts(dt: datetime | None = None) -> str:
    return (dt or datetime.now()).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------- 设备接口定位（SIMULATE 下不依赖仪器库）----------------------

def _ensure_korobka_importable() -> None:
    if importlib.util.find_spec("korobka") is not None:
        return
    for base in (HERE, *HERE.parents):
        for c in (base / "korobka", base):
            if (c / "korobka").is_dir():
                p = str(c)
                if p not in sys.path:
                    sys.path.insert(0, p)
                if importlib.util.find_spec("korobka") is not None:
                    LOG.info("korobka 从 %s 加载", p)
                    return
    raise ImportError("找不到 korobka 包。请用装了 korobka 的环境运行，"
                      "或确认 code 2026/korobka/korobka 存在。")


def _load_transport():
    matches = [m for m in sorted(HERE.glob("**/tool/transport.py")) if "__pycache__" not in str(m)]
    if not matches:
        raise FileNotFoundError(f"在 {HERE} 下找不到 tool/transport.py。请确认脚本放在 data_analysis 目录里。")
    transport_path = matches[-1]
    LOG.info("使用连接模块 %s", transport_path)
    _ensure_korobka_importable()
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("cv_tool_transport", transport_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_cv_script(out_root: Path) -> Path:
    from korobka.cyclic_voltametry.EmStat4X import EmStat4X
    text = EmStat4X.genmscript_CV(
        e_begin_v=E_BEGIN_V, e_vertex1_v=E_VERTEX1_V, e_vertex2_v=E_VERTEX2_V,
        scan_rate_mv_s=SCAN_RATE_MV_S, n_scans=N_SCANS,
        t_equilibrium_s=T_EQUILIBRIUM_S, e_step_mv=E_STEP_MV,
    )
    p = out_root / "methodscript_cv.mscr"
    p.write_text(text, encoding="ascii")
    return p


# ---------------------------------- 单次测量 ----------------------------------

def _count_data_points(data_csv: Path) -> int:
    n = 0
    try:
        with open(data_csv, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                first = line.split(",", 1)[0]
                if first == "":
                    n += 1
                    continue
                try:
                    float(first)
                    n += 1
                except ValueError:
                    continue
    except OSError:
        return 0
    return n


def _simulate_cv(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    time.sleep(SIM_MEAS_SECONDS)
    span = abs(E_VERTEX1_V - E_VERTEX2_V)
    n_per_half = max(2, int(round(span / (E_STEP_MV / 1000.0))))
    rows = []
    for scan in range(N_SCANS):
        up = [E_BEGIN_V + (E_VERTEX1_V - E_BEGIN_V) * k / n_per_half for k in range(n_per_half)]
        dn = [E_VERTEX1_V + (E_VERTEX2_V - E_VERTEX1_V) * k / n_per_half for k in range(n_per_half)]
        for e in up + dn:
            rows.append((scan, e, 1e-6 * math.sin((e - E_BEGIN_V) * 6.0)))
    with open(out_dir / "data.csv", "w", encoding="utf-8") as f:
        f.write("scan,potential,current,time\n")
        for k, (scan, e, cur) in enumerate(rows):
            f.write(f"{scan},{e:.6e},{cur:.6e},{k * DT_S:.6e}\n")
    (out_dir / "raw.txt").write_text("# SIMULATE CV dry-run, no device\n", encoding="ascii")
    return _count_data_points(out_dir / "data.csv")


def _run_one_cv(ctx, out_dir: Path) -> int:
    if SIMULATE:
        return _simulate_cv(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = ctx["transport"].open_device(ctx["port"])
    lines = dev.send_methodscript(str(ctx["cv_script"]), max_duration_s=MAX_MEAS_DURATION_S)
    dev.saveraw_CV(lines, out_dir / "raw.txt")
    dev.savecsv_CV(lines, out_dir / "data.csv", dt=DT_S)
    return _count_data_points(out_dir / "data.csv")


def run_cv_round(ctx):
    """一个 CV 轮次：反复跑 CV，每测完等 GAP_S 秒，持续 ROUND_DURATION_MIN 分钟。"""
    session, round_s = ctx["session"], ROUND_DURATION_MIN * 60.0
    round_start = datetime.now()
    print(f"[CV 轮次] {ROUND_DURATION_MIN} 分钟；每次测完等 {GAP_S}s；满时长不再开新测量。")
    while (datetime.now() - round_start).total_seconds() < round_s:
        ctx["cv_n"] += 1
        out_dir = ctx["out_root"] / f"cv_{ctx['cv_n']:04d}"
        t0 = datetime.now()
        status, n_pts = "ok", ""
        try:
            n_pts = _run_one_cv(ctx, out_dir)
        except Exception as exc:
            LOG.exception("CV 第 %d 次失败", ctx["cv_n"])
            status = f"error: {exc}"
        t1 = datetime.now()
        session.write(event="CV", t_start=_ts(t0), t_end=_ts(t1),
                      elapsed_s=round((t1 - t0).total_seconds(), 3),
                      out_dir=out_dir.name, n_points=n_pts, status=status)
        print(f"  CV[{ctx['cv_n']:04d}] {_ts(t0)} -> {_ts(t1)}  {n_pts} 点  {status}")
        remaining = round_s - (datetime.now() - round_start).total_seconds()
        if remaining <= 0:
            break
        time.sleep(max(0.0, min(GAP_S, remaining)))


# ---------------------------- 共享 session 日志（追加式）----------------------------

SESSION_FIELDS = ["index", "event", "t_start", "t_end", "elapsed_s", "out_dir", "n_points", "status"]


class SessionLog:
    """共享的时间戳日志：文件不存在则新建并写表头，存在则追加。index 接续已有行。
    每写一行即时 flush，中途崩溃也保留已收集记录。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        is_new = (not self.path.exists()) or self.path.stat().st_size == 0
        self._i = 0 if is_new else self._existing_rows()
        self._f = open(self.path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=SESSION_FIELDS)
        if is_new:
            self._w.writeheader()
            self._f.flush()

    def _existing_rows(self) -> int:
        with open(self.path, encoding="utf-8") as f:
            return max(0, sum(1 for _ in f) - 1)   # 去掉表头

    def write(self, event, t_start="", t_end="", elapsed_s="", out_dir="", n_points="", status="ok"):
        self._w.writerow({"index": self._i, "event": event, "t_start": t_start, "t_end": t_end,
                          "elapsed_s": elapsed_s, "out_dir": out_dir, "n_points": n_points, "status": status})
        self._f.flush()
        self._i += 1

    def close(self):
        try:
            self._f.close()
        except OSError:
            pass


def get_session_root(now: datetime) -> Path:
    """当天共享文件夹：analysis_<MMDDYYYY>。存在则复用（不加后缀、不覆盖），不存在则新建。"""
    parent = Path(OUTPUT_PARENT).expanduser().resolve() if OUTPUT_PARENT else HERE
    root = parent / f"{ANALYSIS_PREFIX}_{now.strftime('%m%d%Y')}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def next_index(out_root: Path, prefix: str) -> int:
    """已有 <prefix>_NNNN 子文件夹里的最大编号（没有则 0），新测量从 +1 开始。"""
    nums = []
    for p in out_root.glob(f"{prefix}_*"):
        tail = p.name.split("_", 1)[-1]
        if p.is_dir() and tail.isdigit():
            nums.append(int(tail))
    return max(nums, default=0)


# ---------------------------------- 主流程 ----------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    session_start = datetime.now()
    out_root = get_session_root(session_start)                 # 当天共享文件夹
    session = SessionLog(out_root / "session.csv")             # 共享日志，追加
    cv_start = next_index(out_root, "cv")                      # 本次运行前已有的编号
    ctx = {"transport": None, "port": None, "cv_script": None,
           "out_root": out_root, "session": session,
           "cv_n": cv_start}                                   # 接续已有编号

    mode = "SIMULATE(干跑)" if SIMULATE else "实测"
    print(f"[{mode}] 共享输出目录: {out_root}")
    print(f"CV: {E_BEGIN_V}->{E_VERTEX1_V}->{E_VERTEX2_V} V, {N_SCANS} 圈, "
          f"{SCAN_RATE_MV_S} mV/s, e_step {E_STEP_MV} mV (dt={DT_S:.3f}s), 平衡 {T_EQUILIBRIUM_S}s")

    session.write(event="cv_run_start", t_start=_ts(session_start))

    if not SIMULATE:
        try:
            ctx["transport"] = _load_transport()
            ctx["port"] = ctx["transport"].find_device()
            print(f"已连接 EmStat4X：port={ctx['port']!r}")
            ctx["cv_script"] = _build_cv_script(out_root)
        except Exception as exc:
            LOG.exception("设备初始化失败")
            session.write(event="device_init_failed", t_start=_ts(), status=f"error: {exc}")
            session.close()
            print(f"设备初始化失败：{exc}")
            return 2

    interrupted = False
    try:
        run_cv_round(ctx)
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到中断，正在收尾…")

    session_end = datetime.now()
    total_elapsed = round((session_end - session_start).total_seconds(), 1)
    session.write(event="cv_run_end", t_start=_ts(session_end),
                  elapsed_s=total_elapsed, status="interrupted" if interrupted else "ok")
    session.close()
    print(f"结束：本次 CV {ctx['cv_n'] - cv_start} 次，总用时 {total_elapsed}s。")
    print(f"数据在：{out_root}")
    print(f"共享时间戳日志：{out_root / 'session.csv'}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
