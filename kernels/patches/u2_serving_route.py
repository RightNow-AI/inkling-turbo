#!/usr/bin/env python3
"""U2 serving integration: route sm_90/sm_120 Inkling attention to tml-fa4.

The day-0 router (_use_sheared_bias in fa4_rel_attention.py) sends only
Blackwell (cap 10/11) through tml-fa4's sheared rel_bias path; Hopper and
sm_120 fall back to the score_mod gather, measured 2.6-6.9x slower than our
native sm_90 kernel at the serving anchor shapes (session 24). With the
modified tml_fa4 deployed (native partition_C bias on sm_90, proven generic
path on sm_120), the sheared path is correct on 9/12 too:
  - sm_90 native parity 3/3 on H100 (journal session 24)
  - sm_120 parity 3/3 local (v1)
Requires kernels/tml_fa4_modified deployed first: stock tml-fa4 silently
IGNORES rel_bias on non-Blackwell (upstream finding 01) and would return
wrong output. Idempotent.

Usage: python u2_serving_route.py /path/to/vllm
"""

import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
TARGET = ROOT / "vllm/models/inkling/nvidia/ops/fa4_rel_attention.py"

OLD = """@cache
def _use_sheared_bias() -> bool:
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major in (10, 11)
"""

NEW = """@cache
def _use_sheared_bias() -> bool:
    # Inkling-turbo: sm_90 (native partition_C bias) and sm_120 (generic
    # sheared tile) are parity-proven with the modified tml-fa4 deployed;
    # both beat the score_mod gather at every measured serving shape.
    capability = current_platform.get_device_capability()
    return capability is not None and capability.major in (9, 10, 11, 12)
"""


def main() -> None:
    s = TARGET.read_text()
    if "Inkling-turbo: sm_90" in s:
        print("already applied")
        return
    assert OLD in s, "anchor missing: _use_sheared_bias"
    TARGET.write_text(s.replace(OLD, NEW, 1))
    print("serving route applied: sm_90/sm_120 -> tml-fa4 sheared path")


if __name__ == "__main__":
    main()
