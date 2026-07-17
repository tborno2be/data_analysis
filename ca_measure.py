#!/usr/bin/env python3
"""
CA 测量 —— 测一次计时电流(CA)，只调用 EmStat4X。

做什么：
  运行后先等 PRE_WAIT_S 秒（默认 2 分钟，给你插入样品用），再保持在 CA_POTENTIAL_V
  （默认 0.5V）测一次 CA。默认 1 次；把 N_MEASUREMENTS 改大就连测多次，每次之间等 GAP_S 秒。
  这段插入等待也会记进 session.csv 的时间戳。

时间戳日志（共享）：
  本脚本和 cv_timed_measure.py 当天写进【同一个】 analysis_<日期>/session.csv。
  第一个运行的脚本创建它，之后的（无论 CV 还是 CA）都【追加】进去，
  cv_XXXX / ca_XXXX 的编号自动接着已有的往下排。这样 CV 和 CA 的时间戳在一个文件里。

文件放哪儿：
  本脚本和 cv_timed_measure.py 都放在 data_analysis 根目录（和 analysis_<日期>、tool 平级），
  不要放进 tool/ 里。两个脚本必须在同一目录，才能共享同一个 session。

连接设备用 data_analysis 下 tool/transport.py 的 find_device()/open_device()，底层是
korobka.cyclic_voltametry.EmStat4X。

输出（都在 analysis_<日期>/ 下）：
  ca_0001/ ...   每次 CA：data.csv(time,current，首行记录电位) + raw.txt
  session.csv    共享时间戳日志

怎么跑：
  cd 到 data_analysis，然后  python ca_measure.py
  想不接仪器只验证流程/文件结构，把 SIMULATE 改成 True。
"""

from __future__ import annotations

# =====================================================================
#  参数 —— 在这里填（CA 保持在 0.5V）
# =====================================================================

CA_POTENTIAL_V   = 0.5    # CA 保持电位 (V) —— 从 0.5
CA_INTERVAL_S    = 0.1    # 采样间隔 (s)
CA_DURATION_S    = 60     # 持续时长 (s)

PRE_WAIT_S       = 120    # 插入样品后、开始测量前先等多少秒（默认 2 分钟）；设 0 则不等

N_MEASUREMENTS   = 1      # 测几次（默认 1 次）
GAP_S            = 30     # 连测多次时，每次测完等待多少秒再测下一次

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

LOG = logging.getLogger("ca_measure")
HERE = Path(__file__).resolve().parent


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
    spec = importlib.util.spec_from_file_location("ca_tool_transport", transport_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_ca_script(out_root: Path) -> Path:
    from korobka.cyclic_voltametry.EmStat4X import EmStat4X
    text = EmStat4X.genmscript_CA(
        potential_v=CA_POTENTIAL_V, interval_s=CA_INTERVAL_S, duration_s=CA_DURATION_S,
    )
    p = out_root / "methodscript_ca.mscr"
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


def _simulate_ca(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    time.sleep(SIM_MEAS_SECONDS)
    n = max(2, int(round(CA_DURATION_S / max(CA_INTERVAL_S, 1e-6))))
    with open(out_dir / "data.csv", "w", encoding="utf-8") as f:
        f.write(f"# potential_V={CA_POTENTIAL_V}\n")
        f.write("time,current\n")
        for k in range(n):
            t = k * CA_INTERVAL_S
            f.write(f"{t:.6e},{1e-6 * math.exp(-t / 10.0):.6e}\n")
    (out_dir / "raw.txt").write_text("# SIMULATE CA dry-run, no device\n", encoding="ascii")
    return _count_data_points(out_dir / "data.csv")


def _run_one_ca(ctx, out_dir: Path) -> int:
    if SIMULATE:
        return _simulate_ca(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = ctx["transport"].open_device(ctx["port"])
    lines = dev.send_methodscript(str(ctx["ca_script"]), max_duration_s=MAX_MEAS_DURATION_S)
    dev.saveraw_CA(lines, out_dir / "raw.txt")
    dev.savecsv_CA(lines, out_dir / "data.csv", potential_v=CA_POTENTIAL_V)
    return _count_data_points(out_dir / "data.csv")


def _wait(seconds: float, why: str):
    """睡 seconds 秒，每 30s 打印一次剩余时间。"""
    end = datetime.now().timestamp() + seconds
    print(f"[等待] {why}：{int(seconds)}s …")
    while True:
        remaining = end - datetime.now().timestamp()
        if remaining <= 0:
            break
        time.sleep(min(30.0, remaining))
        left = int(max(0, end - datetime.now().timestamp()))
        if left > 0:
            print(f"        还剩 {left}s")


def run_ca(ctx):
    """插入后等 PRE_WAIT_S 秒，再测 N_MEASUREMENTS 次 CA（保持在 CA_POTENTIAL_V），多次之间等 GAP_S 秒。"""
    session = ctx["session"]

    # 插入样品后、开始测量前的等待
    if PRE_WAIT_S > 0:
        w0 = datetime.now()
        _wait(PRE_WAIT_S, "插入后等待")
        w1 = datetime.now()
        session.write(event=f"wait_{int(PRE_WAIT_S)}s", t_start=_ts(w0), t_end=_ts(w1),
                      elapsed_s=round((w1 - w0).total_seconds(), 1))

    print(f"[CA] 保持 {CA_POTENTIAL_V}V，{CA_DURATION_S}s，间隔 {CA_INTERVAL_S}s，共 {N_MEASUREMENTS} 次。")
    done = 0
    while done < max(1, N_MEASUREMENTS):
        ctx["ca_n"] += 1
        done += 1
        out_dir = ctx["out_root"] / f"ca_{ctx['ca_n']:04d}"
        t0 = datetime.now()
        status, n_pts = "ok", ""
        try:
            n_pts = _run_one_ca(ctx, out_dir)
        except Exception as exc:
            LOG.exception("CA 第 %d 次失败", ctx["ca_n"])
            status = f"error: {exc}"
        t1 = datetime.now()
        session.write(event="CA", t_start=_ts(t0), t_end=_ts(t1),
                      elapsed_s=round((t1 - t0).total_seconds(), 3),
                      out_dir=out_dir.name, n_points=n_pts, status=status)
        print(f"  CA[{ctx['ca_n']:04d}] {_ts(t0)} -> {_ts(t1)}  {n_pts} 点  {status}")
        if done < max(1, N_MEASUREMENTS):
            time.sleep(max(0.0, GAP_S))


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
    ca_start = next_index(out_root, "ca")                      # 本次运行前已有编号
    ctx = {"transport": None, "port": None, "ca_script": None,
           "out_root": out_root, "session": session, "ca_n": ca_start}

    mode = "SIMULATE(干跑)" if SIMULATE else "实测"
    print(f"[{mode}] 共享输出目录: {out_root}")
    print(f"CA: 保持 {CA_POTENTIAL_V}V, {CA_DURATION_S}s, 间隔 {CA_INTERVAL_S}s, 共 {N_MEASUREMENTS} 次")

    session.write(event="ca_run_start", t_start=_ts(session_start))

    if not SIMULATE:
        try:
            ctx["transport"] = _load_transport()
            ctx["port"] = ctx["transport"].find_device()
            print(f"已连接 EmStat4X：port={ctx['port']!r}")
            ctx["ca_script"] = _build_ca_script(out_root)
        except Exception as exc:
            LOG.exception("设备初始化失败")
            session.write(event="device_init_failed", t_start=_ts(), status=f"error: {exc}")
            session.close()
            print(f"设备初始化失败：{exc}")
            return 2

    interrupted = False
    try:
        run_ca(ctx)
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到中断，正在收尾…")

    session_end = datetime.now()
    total_elapsed = round((session_end - session_start).total_seconds(), 1)
    session.write(event="ca_run_end", t_start=_ts(session_end),
                  elapsed_s=total_elapsed, status="interrupted" if interrupted else "ok")
    session.close()
    print(f"结束：本次 CA {ctx['ca_n'] - ca_start} 次，总用时 {total_elapsed}s。")
    print(f"数据在：{out_root}")
    print(f"共享时间戳日志：{out_root / 'session.csv'}")
    return 130 if interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
