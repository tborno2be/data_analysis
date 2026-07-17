"""Cross-platform EmStat4X connection: serial where available, raw USB fallback (macOS)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
import sys
from korobka.cyclic_voltametry.EmStat4X import EmStat4X
from korobka.cyclic_voltametry.exceptions import CommunicationError, DeviceNotFoundError

LOG = logging.getLogger(__name__)

VID, PID = 0x300A, 0x2003

def _usb_backend():
    """Locate a libusb-1.0 backend, checking Homebrew and conda dylib locations."""
    import usb.backend.libusb1 as libusb1
    backend = libusb1.get_backend()
    if backend is not None:
        return backend
    candidates = [
        "/opt/homebrew/lib/libusb-1.0.dylib",
        "/usr/local/lib/libusb-1.0.dylib",
        str(Path(sys.prefix) / "lib" / "libusb-1.0.dylib"),
    ]
    for path in candidates:
        if Path(path).exists():
            backend = libusb1.get_backend(find_library=lambda _x, _p=path: _p)
            if backend is not None:
                LOG.info("libusb backend loaded from %s", path)
                return backend
    raise DeviceNotFoundError("libusb library not found; run 'brew install libusb'.")

def find_device() -> str:
    """Return a serial port name, or 'usb' when only the raw-USB path exists."""
    try:
        return EmStat4X.find_port()
    except DeviceNotFoundError:
        import usb.core
        if usb.core.find(idVendor=VID, idProduct=PID, backend=_usb_backend()) is not None:
            LOG.info("EmStat4X present on USB bus without a serial driver; using raw USB.")
            return "usb"
        raise


def open_device(port: str, baud_rate: int = 921600, timeout: float = 1.0):
    """Open a one-shot measurement connection for the given find_device() result."""
    if port == "usb":
        return USBTransport(timeout=timeout)
    return EmStat4X(port, baud_rate, timeout)


class USBTransport:
    """One-shot EmStat4X connection over raw USB bulk endpoints (macOS fallback)."""

    mock = False
    saveraw_CV = EmStat4X.saveraw_CV
    savecsv_CV = EmStat4X.savecsv_CV
    saveraw_CA = EmStat4X.saveraw_CA
    savecsv_CA = EmStat4X.savecsv_CA

    def __init__(self, timeout: float = 1.0) -> None:
        import usb.core
        import usb.util
        self._usb_core, self._usb_util = usb.core, usb.util
        self.timeout = timeout
        self._buf = b""

        dev = usb.core.find(idVendor=VID, idProduct=PID, backend=_usb_backend())
        if dev is None:
            raise DeviceNotFoundError("No EmStat4X on USB bus.")
        dev.set_configuration()

        self._ep_in = self._ep_out = None
        for intf in dev.get_active_configuration():
            for ep in intf:
                if usb.util.endpoint_type(ep.bmAttributes) != usb.util.ENDPOINT_TYPE_BULK:
                    continue
                if usb.util.endpoint_direction(ep.bEndpointAddress) == usb.util.ENDPOINT_IN:
                    self._ep_in = ep
                else:
                    self._ep_out = ep
        if not (self._ep_in and self._ep_out):
            raise DeviceNotFoundError("EmStat4X bulk endpoint pair not found.")
        self._dev = dev

        line_coding = (921600).to_bytes(4, "little") + bytes([0, 0, 8])
        try:
            dev.ctrl_transfer(0x21, 0x20, 0, 0, line_coding)
            dev.ctrl_transfer(0x21, 0x22, 0x03, 0, None)
        except self._usb_core.USBError:
            pass
        self.flush_buffers()
        LOG.info("EmStat4X opened over raw USB.")

    def flush_buffers(self) -> None:
        """Drain stale bytes left over from previous sessions."""
        self._buf = b""
        while True:
            try:
                if not self._ep_in.read(512, timeout=100):
                    return
            except self._usb_core.USBError:
                return

    def _read_chunk(self, timeout_ms: int) -> bytes:
        try:
            return bytes(self._ep_in.read(512, timeout=timeout_ms))
        except self._usb_core.USBError:
            return b""

    def readline(self) -> bytes:
        deadline = time.monotonic() + self.timeout
        while b"\n" not in self._buf:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                out, self._buf = self._buf, b""
                return out
            self._buf += self._read_chunk(max(1, int(remaining * 1000)))
        line, self._buf = self._buf.split(b"\n", 1)
        return line + b"\n"

    def send_methodscript(self, methodscript_path: str | Path, max_duration_s: float = 480) -> list[str]:
        """Same contract as EmStat4X.send_methodscript, over USB."""
        with open(methodscript_path, encoding="ascii") as f:
            script_text = f.read()
        try:
            self._ep_out.write(script_text.encode("ascii"))
            result_lines: list[str] = []
            start = time.monotonic()
            while True:
                if time.monotonic() - start > max_duration_s:
                    LOG.warning("Timeout after %ss; returning %d line(s).", max_duration_s, len(result_lines))
                    break
                line = self.readline().decode("ascii", errors="replace")
                if line == "":
                    continue
                if line == "\n":
                    break
                if line[-1] != "\n":
                    raise CommunicationError("No EOL character received.")
                result_lines.append(line)
            LOG.info("Measurement finished, %d line(s) received.", len(result_lines))
            return result_lines
        finally:
            self.close()

    def close(self) -> None:
        if getattr(self, "_dev", None) is not None:
            self._usb_util.dispose_resources(self._dev)
            self._dev = None
            LOG.info("USB device released.")