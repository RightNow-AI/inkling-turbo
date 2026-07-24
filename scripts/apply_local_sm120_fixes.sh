#!/usr/bin/env bash
# Fixes required to run the day-0 Inkling FA4 path on sm_120 (RTX 50-series)
# with the fork-pinned toolchain (nvidia-cutlass-dsl 4.6.0). All three are
# upstream-reportable bugs; evidence in journal/local-tier-bringup.md.
#
# Usage: apply_local_sm120_fixes.sh <vllm-checkout-with-installed-third_party>
set -euo pipefail
VLLM_DIR="${1:?usage: $0 <vllm dir>}"

# 1. tml-fa4 @13374f0c written against pre-4.6.0 CuTe DSL: ThrMma/TiledMma
#    moved from cutlass.cute.core to cutlass.cute (verified in 4.6.0
#    site-packages: class ThrMma in cutlass/cute/atom.py, exported at cute.*).
#    Also: cute.make_fragment renamed to cute.make_rmem_tensor (same
#    positional signature; proven on H100 session 1 + local sm_120 parity).
sed -i 's/cute\.core\.ThrMma/cute.ThrMma/g; s/cute\.core\.TiledMma/cute.TiledMma/g; s/cute\.make_fragment(/cute.make_rmem_tensor(/g' \
  "$VLLM_DIR"/vllm/third_party/tml_fa4/*.py "$VLLM_DIR"/vllm/vllm_flash_attn/cute/*.py

# 1b. tml-fa4 keys old-vs-new nvvm API off CUDA 12.9; binding signature
#     tracks nvidia-cutlass-dsl (pinned 4.6.0 = new API). H100 session 2.
python3 - "$VLLM_DIR" <<'PYEOF'
import glob, sys
old = "if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:"
new = "if False:  # nvvm API tracks nvidia-cutlass-dsl (pinned 4.6.0 = new API)"
for p in glob.glob(sys.argv[1] + "/vllm/third_party/tml_fa4/*.py"):
    s = open(p).read()
    if old in s:
        open(p, "w").write(s.replace(old, new))
PYEOF

# 2. vllm_flash_attn cute flash_fwd.py: kernel body reads mDynamicCausal but it
#    was never threaded through the @cute.kernel signature or the launch call
#    (only the generic path is affected, sm90/sm100 have their own files, so
#    CI on H100/B200 never sees it; sm_120 falls back to the generic path).
python3 - "$VLLM_DIR/vllm/vllm_flash_attn/cute/flash_fwd.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
if "mDynamicCausal,\n        ).launch(" not in s:
    s = s.replace(
        "            output_scale,\n        ).launch(",
        "            output_scale,\n            mDynamicCausal,\n        ).launch(")
if "mDynamicCausal: Optional[cute.Tensor] = None,\n    ):\n        # Thread index" not in s:
    s = s.replace(
        "        output_scale: Optional[cute.Tensor] = None,\n    ):\n        # Thread index",
        "        output_scale: Optional[cute.Tensor] = None,\n"
        "        mDynamicCausal: Optional[cute.Tensor] = None,\n"
        "    ):\n        # Thread index")
open(p, "w").write(s)
EOF

# 3. FlashAttentionForwardSm120 (SM80-family shim): shared epilogue in
#    flash_fwd.py reads self.is_split_kv which the SM80-family ctor never sets.
python3 - "$VLLM_DIR/vllm/vllm_flash_attn/cute/flash_fwd_sm120.py" <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
if "self.is_split_kv" not in s:
    s = s.replace(
        "        self.arch = Arch.sm_80\n",
        "        self.arch = Arch.sm_80\n"
        "        if not hasattr(self, \"is_split_kv\"):\n"
        "            self.is_split_kv = False\n")
open(p, "w").write(s)
EOF

echo "sm_120 local fixes applied to $VLLM_DIR"
