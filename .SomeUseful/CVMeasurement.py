"""
EmStat4X cyclic voltammetry driver (CV + CSV only, no Instrument layer).

This version talks to the device directly through BaseSerial:
  - open the port (explicit 8-N-1, short timeout)
  - send the whole MethodSCRIPT in one write (send_binary_command, read_response=False)
  - read result lines one by one until the terminating empty line
  - parse with palmsens.mscript and save raw + CSV

Result success is judged by inspecting the returned/saved raw output: a normal
run ends with the terminating empty line; an error or stall shows up as missing
or truncated raw data.
"""

import time
import logging
from pathlib import Path

import serial   # for the explicit framing constants (PARITY_NONE, etc.)

# Local imports
from korobka.connections.base_serial import BaseSerial
import SomeUseful.palmsens.mscript


LOG = logging.getLogger(__name__)


class CommunicationError(Exception):
	"""Raised when a response line is malformed (no EOL character)."""


class EmStat4X():
	"""Driver class for the PalmSens EmStat4X potentiostat (CV + CSV only)."""

	def __init__(self, device_port="COM4", baud_rate=921600, timeout=1, mock=False):
		self.port     = device_port
		self.baudrate = baud_rate
		self.timeout  = timeout      # short read timeout, paired with the readline loop
		self.mock     = mock

		if self.baudrate is None:
			LOG.error("Baud rate must be provided")
			raise ValueError("Baud rate must be provided")

	# ──────────────────────────────────────────────────────────
	# Communication
	# ──────────────────────────────────────────────────────────

	def send_methodscript(self, methodscript_path, max_duration_s=480):
		r"""Open the port, send a MethodSCRIPT file, and read result lines until
		the terminating empty line.

		Always returns whatever was collected (a list of str lines, each ending
		in '\n'), so the caller can inspect the raw output to judge success.

		:param max_duration_s: hard cap on total wait. If no terminating empty
		    line arrives within this time, the loop stops and returns whatever
		    was collected so far (it does not raise).
		"""
		if self.mock:
			LOG.info("Mock mode: skipping device")
			return ["mock_line\n"]

		# Read the MethodSCRIPT (ASCII only).
		with open(methodscript_path, "rt", encoding="ascii") as f:
			script_text = f.read()

		# Open the port. BaseSerial opens on construction; PalmSens needs 8-N-1,
		# which differs from BaseSerial's 8-E-1 default, so pass it explicitly.
		ser = BaseSerial(
			port=self.port,
			baudrate=self.baudrate,
			bytesize=serial.EIGHTBITS,
			parity=serial.PARITY_NONE,     # 8-N-1  (key: must match the device)
			stopbits=serial.STOPBITS_ONE,
			timeout=self.timeout,          # short timeout, so readline returns
			rtscts=False,                  #   quickly when no data is pending
		)
		try:
			# Send the whole script in one write, without reading a response.
			ser.send_binary_command(script_text.encode("ascii"), read_response=False)

			# Read result lines until the terminating empty line.
			result_lines = []
			start = time.monotonic()
			while True:
				# Overall timeout guard: stop and return what we have (no raise),
				# so a stall can never hang forever -- you still get the raw data.
				if time.monotonic() - start > max_duration_s:
					LOG.warning(
						"Timeout after %ss; returning %d line(s) collected so far.",
						max_duration_s, len(result_lines),
					)
					break

				raw = ser.readline()                       # bytes
				line = raw.decode("ascii", errors="replace")

				if line == "":                             # read timed out, no data
					continue                               #   still measuring -> wait
				if line == "\n":                           # empty line = end of run
					break

				# ASCII protocol check: a complete line ends with '\n'. If not,
				# the line was truncated (e.g. mid-line timeout / link issue).
				if line[-1] != "\n":
					raise CommunicationError("No EOL character received.")

				result_lines.append(line)

			LOG.info("Measurement finished, %d line(s) received.", len(result_lines))
			return result_lines
		finally:
			ser.close()

	# ──────────────────────────────────────────────────────────
	# MethodSCRIPT generation
	# ──────────────────────────────────────────────────────────

	@staticmethod
	def generate_methodscript(
		e_begin_v:       float,
		e_vertex1_v:     float,
		e_vertex2_v:     float,
		scan_rate_mv_s:  float,
		n_scans:         int,
		t_equilibrium_s: int,
		e_step_mv:       float = 5,
	) -> str:
		"""Generate a MethodSCRIPT string for cyclic voltammetry."""
		def fv(v):
			return str(int(round(v * 1000)))

		begin = fv(e_begin_v)
		v1    = fv(e_vertex1_v)
		v2    = fv(e_vertex2_v)
		scan_rate = str(int(round(scan_rate_mv_s)))
		e_step    = str(int(round(e_step_mv)))

		return "\n".join([
			"e",
			"var c",
			"var p",
			"set_pgstat_mode 2",
			"set_max_bandwidth 80",
			f"set_range_minmax da {v1}m {v2}m",
			"set_range ba 590u",
			"set_autoranging ba 590n 590u",
			f"set_e {begin}m",
			"cell_on",
			f"meas_loop_ca p c {begin}m 200m {t_equilibrium_s}",
			"pck_start",
			"    pck_add p",
			"    pck_add c",
			"pck_end",
			"endloop",
			f"meas_loop_cv p c {begin}m {v1}m {v2}m {e_step}m {scan_rate}m nscans({n_scans})",
			"pck_start",
			"    pck_add p",
			"    pck_add c",
			"pck_end",
			"endloop",
			"on_finished:",
			"cell_off",
			"",
			"",
		])

	# ──────────────────────────────────────────────────────────
	# Saving results
	# ──────────────────────────────────────────────────────────

	def save_raw_result(self, result_lines, raw_result_path):
		"""Save raw PalmSens result lines."""
		raw_result_path = Path(raw_result_path)
		raw_result_path.parent.mkdir(parents=True, exist_ok=True)

		with open(raw_result_path, "wt", encoding="ascii") as f:
			f.writelines(result_lines)

		LOG.info("Raw result saved to %s", raw_result_path)
		return raw_result_path

	def save_csv_result(self, result_lines, csv_result_path):
		"""Parse PalmSens result lines and save loop/potential/current as CSV."""
		csv_result_path = Path(csv_result_path)

		if self.mock:
			csv_result_path.parent.mkdir(parents=True, exist_ok=True)
			with open(csv_result_path, "w", encoding="utf-8") as f:
				f.write("loop,potential,current\n0,0.0,0.0\n")
			LOG.info("Mock: CSV saved to %s", csv_result_path)
			return csv_result_path

		csv_result_path.parent.mkdir(parents=True, exist_ok=True)
		curves = SomeUseful.palmsens.mscript.parse_result_lines(result_lines)

		with open(csv_result_path, "w", encoding="utf-8") as f:
			f.write("loop,potential,current\n")
			for iloop, loop in enumerate(curves.loops):
				for potential, current in zip(
					loop.get_column_values(0),
					loop.get_column_values(1),
				):
					f.write(f"{iloop},{potential:.6e},{current:.6e}\n")

		LOG.info("CSV result saved to %s", csv_result_path)
		return csv_result_path

	def log_cv_result(self, result_lines):
		"""Parse and log PalmSens result packages. Returns curves object."""
		if self.mock:
			LOG.info("Mock mode: skipping log")
			return None

		curves = SomeUseful.palmsens.mscript.parse_result_lines(result_lines)
		for loop in curves.loops:
			for package in loop.packages:
				LOG.info([str(v) for v in package])
		LOG.info("CV result packages logged")
		return curves