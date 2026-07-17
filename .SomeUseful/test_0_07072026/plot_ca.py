from pathlib import Path
from korobka.cyclic_voltametry.EmStat4X import EmStat4X

raw_path = Path("test/test_iso_2/Ca1/raw.txt")
lines = raw_path.read_text().splitlines(keepends=True)   # 注意 keepends

em = EmStat4X.__new__(EmStat4X)   # 不走串口初始化，只借这个方法
em.mock = False
em.savecsv_CA(lines, raw_path.parent / "data.csv", potential_v=0.4)