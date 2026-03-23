#!/usr/bin/env python3
"""Pre-build FlashInfer JIT kernels for SM121a before vLLM model loading.

Run this BEFORE vllm server starts so that JIT compilation happens
while GPU memory is free (no model weights loaded).
"""
import os
import sys
import time

os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "12.1a")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

# Use CUDA nvcc for JIT compilation (supports compute_121a)
# Try CUDA 13.1 first, then fall back to default /usr/local/cuda
for cuda_path in ["/usr/local/cuda-13.1", "/usr/local/cuda-13", "/usr/local/cuda"]:
    if os.path.exists(f"{cuda_path}/bin/nvcc"):
        os.environ["CUDA_HOME"] = cuda_path
        print(f"[JIT prebuild] Using CUDA_HOME={cuda_path}")
        break

# JIT prebuild runs before model loading so memory is free — use all CPU cores
os.environ["MAX_JOBS"] = str(os.cpu_count() or 20)

import torch
torch.cuda.set_device(0)

cc = torch.cuda.get_device_capability(0)
print(f"[JIT prebuild] GPU: {torch.cuda.get_device_name(0)}, CC: {cc}")

# Delete cached TVM FFI C DLPack addon so it rebuilds with our patched
# _build_optional_torch_c_dlpack.py (has_storage() fix for SM121)
import glob, shutil
for cached_so in glob.glob(os.path.expanduser("~/.cache/tvm-ffi/libtorch_c_dlpack_addon_*.so")):
    os.remove(cached_so)
    print(f"[JIT prebuild] Deleted cached TVM FFI addon: {cached_so}")

# Delete AOT-compiled fused_moe_120.so — it was built during Docker image build
# with the ORIGINAL cutlass_heuristic.cpp (has CtaShape128x128x128B which exceeds
# SM121's 101KB SMEM). Must force JIT rebuild with our patched heuristic.
# Use a marker file in the persistent cache (~/.cache) to avoid rebuilding every restart.
import flashinfer.jit.env as jit_env
from pathlib import Path
aot_moe_dir = Path(jit_env.FLASHINFER_AOT_DIR) / "fused_moe_120"
marker_file = Path(os.path.expanduser("~/.cache/flashinfer/.moe_patched"))
if aot_moe_dir.exists():
    shutil.rmtree(aot_moe_dir)
    print(f"[JIT prebuild] Deleted AOT fused_moe_120: {aot_moe_dir}")
if not marker_file.exists():
    # First run or cache wiped — delete JIT cache so it rebuilds with patched heuristic
    for fused_moe_cache in glob.glob(os.path.expanduser("~/.cache/flashinfer/*/121a/cached_ops/fused_moe_120")):
        shutil.rmtree(fused_moe_cache)
        print(f"[JIT prebuild] Deleted cached fused_moe module: {fused_moe_cache}")
else:
    print("[JIT prebuild] MoE already patched, using JIT cache")

if cc[0] < 12:
    print("[JIT prebuild] Not SM120+, skipping")
    sys.exit(0)

# Apply CUTLASS SMEM alignment optimization patches
# (reduces alignas(1024)→alignas(128), fixes static_assert for SM120 SMEM)
try:
    CUTLASS_DIR = "/usr/local/lib/python3.12/dist-packages/flashinfer/data/cutlass/include/cutlass"
    _patched = 0
    for fname in ["gemm/collective/sm120_blockscaled_mma_array_tma.hpp",
                   "gemm/collective/sm120_blockscaled_mma_tma.hpp"]:
        fpath = f"{CUTLASS_DIR}/{fname}"
        if os.path.exists(fpath):
            c = open(fpath).read()
            if "alignas(1024)" in c:
                open(fpath, 'w').write(c.replace("alignas(1024)", "alignas(128)"))
                _patched += 1
    # Fix static_assert to use sm120 SMEM capacity instead of sm100
    sa_file = f"{CUTLASS_DIR}/gemm/kernel/sm103_blockscaled_gemm_array_tma_warpspecialized.hpp"
    if os.path.exists(sa_file):
        c = open(sa_file).read()
        old_sa = "static_assert(SharedStorageSize <= cutlass::arch::sm100_smem_capacity_bytes"
        new_sa = "static_assert(SharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes"
        if old_sa in c:
            open(sa_file, 'w').write(c.replace(old_sa, new_sa))
            _patched += 1
    if _patched:
        print(f"[JIT prebuild] Applied {_patched} CUTLASS SMEM alignment patches")
        # Force JIT rebuild when patches applied
        if marker_file.exists():
            marker_file.unlink()
            for fused_moe_cache in glob.glob(os.path.expanduser("~/.cache/flashinfer/*/121a/cached_ops/fused_moe_120")):
                shutil.rmtree(fused_moe_cache)
                print(f"[JIT prebuild] Cleared MoE cache for rebuild with new patches")
except Exception as e:
    print(f"[JIT prebuild] CUTLASS patch warning: {e}")

start = time.time()

# 1. FP4 GEMM CUTLASS SM120
print("[JIT prebuild] 1/4 fp4_gemm_cutlass_sm120...")
from flashinfer.jit.gemm.core import gen_gemm_sm120_module_cutlass_fp4
gen_gemm_sm120_module_cutlass_fp4().build_and_load()

# 2. FP4 quantization
print("[JIT prebuild] 2/5 fp4_quantization...")
try:
    from flashinfer import nvfp4_quantize, SfLayout
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    nvfp4_quantize(x, torch.tensor(100.0, device="cuda"),
                   sfLayout=SfLayout.layout_128x4, do_shuffle=False)
    torch.cuda.synchronize()
except Exception as e:
    print(f"  Skipped (non-critical): {e}")

# 3. Dense GEMM SM120
print("[JIT prebuild] 3/5 gemm_sm120 (dense)...")
from flashinfer.jit.gemm.core import gen_gemm_sm120_module
gen_gemm_sm120_module().build_and_load()

# 4. Fused MoE CUTLASS SM120 (grouped GEMM — longest build)
print("[JIT prebuild] 4/5 fused_moe_sm120 (CUTLASS grouped GEMM)...")
from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm120_module
gen_cutlass_fused_moe_sm120_module(use_fast_build=False).build_and_load()
marker_file.parent.mkdir(parents=True, exist_ok=True)
marker_file.touch()
print(f"[JIT prebuild] Created marker: {marker_file}")

# 5. Attention (TRT-LLM FMHA v2)
print("[JIT prebuild] 5/5 trtllm_fmha_v2...")
try:
    from flashinfer.jit.attention.modules import gen_trtllm_fmha_v2_module
    gen_trtllm_fmha_v2_module().build_and_load()
except Exception as e:
    print(f"  Skipped (non-critical): {e}")

torch.cuda.synchronize()
elapsed = time.time() - start
print(f"[JIT prebuild] Done in {elapsed:.1f}s")

# Free GPU memory
del x
torch.cuda.empty_cache()
