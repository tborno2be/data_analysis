"""Probe the EmStat4X over raw USB: dump descriptors, then try the 't' handshake."""

import usb.core
import usb.backend.libusb1 as libusb1_backend
_backend = libusb1_backend.get_backend(find_library=lambda x: "/opt/homebrew/lib/libusb-1.0.dylib")

VID, PID = 0x300A, 0x2003  # PalmSens EmStat4X, from ioreg (12298 / 8195)

dev = usb.core.find(idVendor=VID, idProduct=PID)
if dev is None:
    raise SystemExit("EmStat4X not found on USB bus.")

dev.set_configuration()
cfg = dev.get_active_configuration()

ep_out = ep_in = None
for intf in cfg:
    print(f"Interface {intf.bInterfaceNumber}: class=0x{intf.bInterfaceClass:02x} "
          f"subclass=0x{intf.bInterfaceSubClass:02x} protocol=0x{intf.bInterfaceProtocol:02x}")
    for ep in intf:
        kind = usb.util.endpoint_type(ep.bmAttributes)
        direction = usb.util.endpoint_direction(ep.bEndpointAddress)
        print(f"  endpoint 0x{ep.bEndpointAddress:02x} type={kind} dir={'IN' if direction else 'OUT'}")
        if kind == usb.util.ENDPOINT_TYPE_BULK:
            if direction == usb.util.ENDPOINT_IN:
                ep_in = ep
            else:
                ep_out = ep

if not (ep_in and ep_out):
    raise SystemExit("No bulk endpoint pair found.")

# CDC housekeeping: SET_LINE_CODING (921600 8N1) + SET_CONTROL_LINE_STATE (DTR|RTS).
# Many CDC firmwares ignore these, some refuse to talk without them; sending both is harmless.
line_coding = (921600).to_bytes(4, "little") + bytes([0, 0, 8])
try:
    dev.ctrl_transfer(0x21, 0x20, 0, 0, line_coding)
    dev.ctrl_transfer(0x21, 0x22, 0x03, 0, None)
except usb.core.USBError as exc:
    print(f"CDC control transfers not accepted ({exc}); continuing anyway.")

ep_out.write(b"t\n")
buf = b""
for _ in range(20):                      # up to ~4 s total
    try:
        buf += bytes(ep_in.read(256, timeout=200))
    except usb.core.USBError:            # timeout on an idle pipe: response finished
        if buf:
            break
print(f"Full response: {buf!r}")
print("MATCH" if b"tes4" in buf else "no tes4 anywhere")