"""Reproduce the exact bug that lost an 8x B200: remote output containing a
byte undecodable in the Windows default codepage (cp1252) killed the
subprocess reader thread, leaving CompletedProcess.stdout as None."""
import subprocess
import sys

CHILD = (
    "import sys; sys.stdout.buffer.write(b'ok\\x8f\\xffdone'); "
    "sys.stdout.buffer.flush()"
)

print("interpreter:", sys.executable)

try:
    old = subprocess.run([sys.executable, "-c", CHILD], capture_output=True,
                         text=True, encoding="cp1252", timeout=30)
    print("cp1252 stdout :", repr(old.stdout))
except UnicodeDecodeError as exc:
    print("cp1252 RAISED :", type(exc).__name__, exc)

new = subprocess.run([sys.executable, "-c", CHILD], capture_output=True,
                     text=True, encoding="utf-8", errors="replace", timeout=30)
# the console here is cp1252; encode defensively so printing the result
# cannot itself raise (that would mask the thing we are proving)
print("utf8+replace  :", ascii(new.stdout))

assert new.stdout is not None, "stdout is None: hardening FAILED"
assert "ok" in new.stdout and "done" in new.stdout, "content lost: FAILED"
print("PASS: undecodable bytes degrade to replacement chars, stream intact")
