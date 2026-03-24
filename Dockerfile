# syntax=docker/dockerfile:1.6

# Limit build parallelism to reduce OOM situations
ARG BUILD_JOBS=8

# =========================================================
# STAGE 1: Base Image (Installs Dependencies)
# =========================================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS base

# Build parallemism
ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

# Set non-interactive frontend to prevent apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Allow pip to install globally on Ubuntu 24.04 without a venv
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set pip cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# Set the base directory environment variable
ENV VLLM_BASE_DIR=/workspace/vllm

# 1. Install Build Dependencies & Ccache
# Added ccache to enable incremental compilation caching
RUN apt update && \
    apt install -y --no-install-recommends \
    curl vim ninja-build git \
    ccache \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv && pip uninstall -y flash-attn

# Configure Ccache for CUDA/C++
ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
# Limit ccache size to prevent unbounded growth (e.g. 50G)
ENV CCACHE_MAXSIZE=50G
# Enable compression to save space
ENV CCACHE_COMPRESS=1
# Tell CMake to use ccache for compilation
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# Setup Workspace
WORKDIR $VLLM_BASE_DIR

# 2. Set Environment Variables
ARG TORCH_CUDA_ARCH_LIST="12.0a;12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# =========================================================
# STAGE 2: FlashInfer Builder
# =========================================================
FROM base AS flashinfer-builder

ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
WORKDIR $VLLM_BASE_DIR
ARG FLASHINFER_REF=main

# --- CACHE BUSTER ---
# Change this argument to force a re-download of FlashInfer
ARG CACHEBUST_FLASHINFER=1

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# Smart Git Clone (Fetch changes instead of full re-clone)
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "flashinfer" ]; then \
        echo "Cache miss: Cloning FlashInfer from scratch..." && \
        git clone --recursive https://github.com/flashinfer-ai/flashinfer.git; \
        if [ "$FLASHINFER_REF" != "main" ]; then \
            cd flashinfer && \
            git checkout ${FLASHINFER_REF}; \
        fi; \
    else \
        echo "Cache hit: Fetching flashinfer updates..." && \
        cd flashinfer && \
        git fetch origin && \
        git fetch origin --tags --force && \
        (git checkout --detach origin/${FLASHINFER_REF} 2>/dev/null || git checkout ${FLASHINFER_REF}) && \
        git submodule update --init --recursive && \
        git clean -fdx && \
        git gc --auto; \
    fi && \
    cp -a /repo-cache/flashinfer /workspace/flashinfer

WORKDIR /workspace/flashinfer

# Bump CUTLASS to v4.4.2 — fixes grouped GEMM SMEM stage count (PR #3092),
# TMA descriptor alignment (#2905/#2906), and zero-stride TMA basis.
# Reference: https://github.com/NVIDIA/cutlass/issues/3096
RUN cd 3rdparty/cutlass && \
    git fetch origin && \
    git checkout v4.4.2 && \
    cd ../..

# Apply K=64 SM120 block-scaled MoE GEMM patch (for CUTLASS v4.4.2)
# Enables 7-11 pipeline stages vs 2 with K=128, giving ~2x decode throughput.
# - EffBlk_SF clamping in sm120_blockscaled_mma_builder.inl
# - K=64 tile shapes in are_tile_shapes_supported_sm120
# - K=64 CTA shapes in generate_kernels.py
# Reference: https://github.com/flashinfer-ai/flashinfer/pull/2786
#            https://github.com/NVIDIA/cutlass/issues/3096
COPY patches/build/flashinfer_k64_sm120_v442.patch .
RUN patch -p1 < flashinfer_k64_sm120_v442.patch

# Remove K=128 large tiles that don't fit SM121's 101KB SMEM (Stages=1 → static_assert fail).
# Keep only K=64 tiles + 128x128x128 (the only K=128 tile that fits with 2 stages).
RUN sed -i 's/\[128, 128, 256\],/# REMOVED for SM121: [128, 128, 256],/' flashinfer/jit/gemm/cutlass/generate_kernels.py && \
    sed -i 's/\[128, 256, 128\],/# REMOVED for SM121: [128, 256, 128],/' flashinfer/jit/gemm/cutlass/generate_kernels.py && \
    sed -i 's/\[256, 128, 128\],/# REMOVED for SM121: [256, 128, 128],/' flashinfer/jit/gemm/cutlass/generate_kernels.py && \
    echo "[OK] Removed large K=128 tiles from generate_kernels.py" && \
    sed -i '/TileM == 128 && TileN == 128 && TileK == 256/d' \
        csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h && \
    sed -i '/TileM == 128 && TileN == 256 && TileK == 128/d' \
        csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h && \
    sed -i '/TileM == 256 && TileN == 128 && TileK == 128/d' \
        csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_gemm_template_dispatch_tma_ws.h && \
    echo "[OK] Removed large K=128 tiles from dispatch header"

# Enable GDC (Grid Dependency Control) for SM100+ in ALL FlashInfer compilation paths.
# Without this, PDL barriers (griddepcontrol.wait/launch_dependents) compile as no-ops,
# causing race conditions between dependent MoE kernels → illegal instruction during
# CUDA graph capture. The env var is picked up by build_cuda_cflags() in cpp_ext.py,
# which covers both the JIT path (via core.py) AND the AOT path (fused_moe, fp4_quantization)
# that uses CompilationContext.get_nvcc_flags_list() — a separate code path that the
# previous core.py sed didn't reach.
# Reference: FlashInfer PR #2780
ENV FLASHINFER_EXTRA_CUDAFLAGS="-DCUTLASS_ENABLE_GDC_FOR_SM100=1"

ARG FLASHINFER_PRS=""

RUN if [ -n "$FLASHINFER_PRS" ]; then \
        echo "Applying FlashInfer PRs: $FLASHINFER_PRS"; \
        for pr in $FLASHINFER_PRS; do \
            echo "Fetching and applying FlashInfer PR #$pr..."; \
            curl -fL "https://github.com/flashinfer-ai/flashinfer/pull/${pr}.diff" | git apply -v; \
        done; \
    fi

# Apply E2M1 SM121 fix: remove SM121 from CUDA_PTX_FP4FP6_CVT_ENABLED
# SM121 (GB10) lacks cvt.rn.satfinite.e2m1x2.f32 PTX instruction
# Reference: https://github.com/Avarok-Cybersecurity/dgx-vllm
COPY patches/build/flashinfer_e2m1_sm121.patch .
RUN if [ -f flashinfer_e2m1_sm121.patch ]; then \
        if patch -p1 --dry-run --reverse < flashinfer_e2m1_sm121.patch &>/dev/null; then \
            echo "E2M1 SM121 CUTLASS patch already applied"; \
        else \
            echo "Applying E2M1 SM121 CUTLASS patch..." && \
            patch -p1 < flashinfer_e2m1_sm121.patch; \
        fi; \
    fi

# Fix TRT-LLM quantization_utils.cuh for SM121: add software E2M1 conversion
# fallback ONLY in the low-level fp32_vec_to_e2m1 functions.
# IMPORTANT: Do NOT globally exclude SM121 from all >= 1000 guards — the
# higher-level wrapper functions (cvt_warp_fp16_to_fp4, etc.) must be allowed
# to enter the >= 1000 path on SM121. They do generic float math and call
# fp32_vec_to_e2m1 which has the software fallback. Excluding SM121 from
# the wrappers causes them to return 0 with uninitialized scale factors → NaN.
# Reference: https://github.com/Avarok-Cybersecurity/dgx-vllm
COPY patches/build/fix_quantization_utils_sm121.py .
RUN python3 fix_quantization_utils_sm121.py

# Patch CuTe DSL admissible_archs: add SM120/SM121 so MoE JIT can compile for GB10
RUN for f in 3rdparty/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/mma.py \
           3rdparty/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/tcgen05/copy.py \
           3rdparty/cutlass/python/CuTeDSL/cutlass/cute/nvgpu/cpasync/copy.py; do \
        if [ -f "$f" ] && ! grep -q 'sm_121a' "$f"; then \
            sed -i 's/"sm_100f",/"sm_100f",\n        "sm_120a", "sm_120f", "sm_121a",/' "$f" && \
            echo "  [OK] Patched $f"; \
        else \
            echo "  [SKIP] $f (already patched or not found)"; \
        fi; \
    done

# Patch TRT-LLM EpilogueSubTile: == 100 → >= 100 (PR #5823) for SM120/SM121
RUN if grep -q 'kMinComputeCapability == 100' \
       csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl 2>/dev/null; then \
        sed -i 's/kMinComputeCapability == 100/kMinComputeCapability >= 100/g' \
            csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl && \
        echo "TRT-LLM PR #5823: == 100 -> >= 100"; \
    else \
        echo "TRT-LLM PR #5823: already applied or file not found"; \
    fi

# Apply patch to avoid re-downloading existing cubins
COPY patches/build/flashinfer_cache.patch .
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=cubins-cache,target=/workspace/flashinfer/flashinfer-cubin/flashinfer_cubin/cubins \
    patch -p1 < flashinfer_cache.patch && \
    # flashinfer-python
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' -e '/license-files/d' pyproject.toml && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    # flashinfer-cubin
    cd flashinfer-cubin && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    # flashinfer-jit-cache
    cd ../flashinfer-jit-cache && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    # dump git ref in the wheels dir
    cd .. && git rev-parse HEAD > /workspace/wheels/.flashinfer-commit

# =========================================================
# STAGE 3: FlashInfer Wheel Export
# =========================================================
FROM scratch AS flashinfer-export
COPY --from=flashinfer-builder /workspace/wheels /

# =========================================================
# STAGE 4: vLLM Builder
# =========================================================
FROM base AS vllm-builder

ARG TORCH_CUDA_ARCH_LIST="12.0a;12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
WORKDIR $VLLM_BASE_DIR

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# --- VLLM SOURCE CACHE BUSTER ---
ARG CACHEBUST_VLLM=1

# Git reference (branch, tag, or SHA) to checkout
# Pin to the commit we tested against. Use --build-arg VLLM_REF=main for latest.
ARG VLLM_REF=ed359c497

# Smart Git Clone (Fetch changes instead of full re-clone)
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "vllm" ]; then \
        echo "Cache miss: Cloning vLLM from scratch..." && \
        git clone --recursive https://github.com/vllm-project/vllm.git; \
        if [ "$VLLM_REF" != "main" ]; then \
            cd vllm && \
            git checkout ${VLLM_REF}; \
        fi; \
    else \
        echo "Cache hit: Fetching updates..." && \
        cd vllm && \
        git fetch origin && \
        git fetch origin --tags --force && \
        (git checkout --detach origin/${VLLM_REF} 2>/dev/null || git checkout ${VLLM_REF}) && \
        git submodule update --init --recursive && \
        git clean -fdx && \
        git gc --auto; \
    fi && \
    cp -a /repo-cache/vllm $VLLM_BASE_DIR/

WORKDIR $VLLM_BASE_DIR/vllm

ARG VLLM_PRS=""

RUN if [ -n "$VLLM_PRS" ]; then \
        echo "Applying PRs: $VLLM_PRS"; \
        for pr in $VLLM_PRS; do \
            echo "Fetching and applying PR #$pr..."; \
            curl -fL "https://github.com/vllm-project/vllm/pull/${pr}.diff" | git apply -v; \
        done; \
    fi

# Prepare build requirements
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/test.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test.txt && \
    uv pip install -r requirements/build.txt

# Apply Patches
# TEMPORARY PATCH for fastsafetensors loading in cluster setup - tracking https://github.com/vllm-project/vllm/issues/34180
# COPY fastsafetensors.patch .
# RUN if patch -p1 --dry-run --reverse < fastsafetensors.patch &>/dev/null; then \
#         echo "PR #34180 is already applied"; \
#     else \
#         patch -p1 < fastsafetensors.patch; \
#     fi
# TEMPORARY PATCH for broken vLLM build (unguarded Hopper code) - reverting PR #34758 and #34302
RUN curl -L https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/34758.diff | patch -p1 -R || echo "Cannot revert PR #34758, skipping"
RUN curl -L https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/34302.diff | patch -p1 -R || echo "Cannot revert PR #34302, skipping"

# Disable libtorch_stable: incompatible with PyTorch in NGC 26.01 (stableivalue_conversions.h)
# Not needed for inference serving on DGX Spark.
# Remove from both CMakeLists.txt and setup.py
COPY <<'DISABLE_STABLE' /tmp/disable_libtorch_stable.py
# 1. Comment out CMake block containing VLLM_STABLE_EXT_SRC
with open('CMakeLists.txt') as f:
    lines = f.readlines()
start = None
for i, line in enumerate(lines):
    if 'VLLM_STABLE_EXT_SRC' in line:
        for j in range(i, -1, -1):
            if lines[j].strip().startswith('if('):
                start = j
                break
        break
if start is not None:
    depth = 0
    end = start
    for i in range(start, len(lines)):
        s = lines[i].strip()
        if s.startswith('if(') or s.startswith('if ('):
            depth += 1
        if s == 'endif()':
            depth -= 1
            if depth == 0:
                end = i
                break
    for i in range(start, end + 1):
        lines[i] = '# DISABLED: ' + lines[i]
    with open('CMakeLists.txt', 'w') as f:
        f.writelines(lines)
    print(f'[OK] CMakeLists.txt: disabled libtorch_stable block (lines {start+1}-{end+1})')

# 2. Remove _C_stable_libtorch from setup.py
with open('setup.py') as f:
    c = f.read()
c = c.replace('        ext_modules.append(CMakeExtension(name="vllm._C_stable_libtorch"))\n', '        pass  # _C_stable_libtorch disabled\n')
c = c.replace('                    "vllm/_C_stable_libtorch.abi3.so",\n', '')
with open('setup.py', 'w') as f:
    f.write(c)
print('[OK] setup.py: removed _C_stable_libtorch references')

# 3. Remove import from vllm/platforms/cuda.py
with open('vllm/platforms/cuda.py') as f:
    c = f.read()
c = c.replace('import vllm._C_stable_libtorch  # noqa\n', '# import vllm._C_stable_libtorch  # disabled for NGC 26.01\n')
c = c.replace('import vllm._C_stable_libtorch  # noqa', '# import vllm._C_stable_libtorch  # disabled for NGC 26.01')
with open('vllm/platforms/cuda.py', 'w') as f:
    f.write(c)
print('[OK] cuda.py: disabled _C_stable_libtorch import')
DISABLE_STABLE
RUN python3 /tmp/disable_libtorch_stable.py

# Add SM121 to vLLM's CUDA_SUPPORTED_ARCHS so the build emits sm_121a code.
# Without this, vLLM's cmake/utils.cmake maps 12.1a → 12.0 (nearest supported),
# and the E2M1 software fallback (#if __CUDA_ARCH__ == 1210) never fires.
RUN sed -i 's/set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0")/set(CUDA_SUPPORTED_ARCHS "7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0;12.1")/' CMakeLists.txt && \
    sed -i 's/set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0")/set(CUDA_SUPPORTED_ARCHS "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0;12.1")/' CMakeLists.txt

# Apply E2M1 software conversion for SM121 (GB10) - enables CUDA graphs with NVFP4
# SM121 lacks cvt.rn.satfinite.e2m1x2.f32 PTX; this adds a software fallback
# Reference: https://github.com/Avarok-Cybersecurity/dgx-vllm
COPY patches/build/e2m1_nvfp4_sm121.patch .
RUN if [ -f e2m1_nvfp4_sm121.patch ]; then \
        if patch -p1 --dry-run --reverse < e2m1_nvfp4_sm121.patch &>/dev/null; then \
            echo "E2M1 NVFP4 SM121 patch already applied"; \
        else \
            echo "Applying E2M1 NVFP4 SM121 patch..." && \
            patch -p1 < e2m1_nvfp4_sm121.patch; \
        fi; \
    fi

# Patch FP8 ScaledMM: reject cc>=120 (FlashInfer & CUTLASS FP8 lack SM120 kernel images)
COPY <<'PATCH_FP8' /tmp/patch_fp8_sm120.py
import py_compile, sys

# 1. FlashInfer FP8
path1 = 'vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py'
try:
    with open(path1) as f:
        c = f.read()
    if 'compute_capability >= 120' not in c and 'compute_capability < 100' in c:
        c = c.replace('compute_capability < 100', 'compute_capability < 100 or compute_capability >= 120')
        c = c.replace('requires compute capability 100 and above', 'requires compute capability 100-119')
        with open(path1, 'w') as f:
            f.write(c)
        py_compile.compile(path1, doraise=True)
        print('[OK] FlashInfer FP8 patched')
    else:
        print('[SKIP] FlashInfer FP8 already patched or pattern not found')
except FileNotFoundError:
    print(f'[SKIP] {path1} not found')

# 2. CUTLASS FP8
path2 = 'vllm/model_executor/kernels/linear/scaled_mm/cutlass.py'
try:
    with open(path2) as f:
        c = f.read()
    if 'CUTLASS FP8 lacks sm120' not in c:
        old = '        if not current_platform.is_cuda():\n            return False, "requires CUDA."\n        return True, None'
        if c.count(old) >= 2:
            idx1 = c.index(old)
            idx2 = c.index(old, idx1 + 1)
            new = '        if not current_platform.is_cuda():\n            return False, "requires CUDA."\n        if compute_capability is not None and compute_capability >= 120:\n            return False, "CUTLASS FP8 lacks sm120 kernel images."\n        return True, None'
            c = c[:idx2] + c[idx2:].replace(old, new, 1)
            with open(path2, 'w') as f:
                f.write(c)
            py_compile.compile(path2, doraise=True)
            print('[OK] CUTLASS FP8 patched')
        else:
            print('[SKIP] CUTLASS FP8 pattern not found (may have changed)')
    else:
        print('[SKIP] CUTLASS FP8 already patched')
except FileNotFoundError:
    print(f'[SKIP] {path2} not found')
PATCH_FP8
RUN python3 /tmp/patch_fp8_sm120.py

# Patch Mamba SM12x Triton sync workaround (vllm#37431)
# Triton PTX codegen for SM12x has missing memory barriers causing async crashes.
COPY <<'PATCH_MAMBA' /tmp/patch_mamba_sync.py
import py_compile

path = 'vllm/model_executor/layers/mamba/mamba_mixer2.py'
try:
    with open(path) as f:
        content = f.read()
except FileNotFoundError:
    print(f'[SKIP] {path} not found')
    exit(0)

if '_needs_triton_sync' in content:
    print('[SKIP] Mamba sync already patched')
    exit(0)

# 1. Add _needs_triton_sync flag after is_blackwell
old_init = "        self.is_blackwell = current_platform.is_device_capability_family(100)"
new_init = """        self.is_blackwell = current_platform.is_device_capability_family(100)
        # SM12x Triton workaround (vllm#37431): Triton PTX codegen on SM12x
        # has missing memory barriers causing async execution crashes.
        self._needs_triton_sync = current_platform.is_device_capability_family(120)"""
assert old_init in content, f'is_blackwell init not found in {path}'
content = content.replace(old_init, new_init)

# 2. Add sync at start of conv_ssm_forward
old_csf = """\
    def conv_ssm_forward(
        self,
        projected_states: torch.Tensor,
        output: torch.Tensor,
    ):"""
new_csf = """\
    def conv_ssm_forward(
        self,
        projected_states: torch.Tensor,
        output: torch.Tensor,
    ):
        # SM12x Triton workaround: sync before Mamba Triton kernels (vllm#37431)
        if self._needs_triton_sync:
            torch.cuda.synchronize()"""
assert old_csf in content, f'conv_ssm_forward signature not found in {path}'
content = content.replace(old_csf, new_csf)

# 3. Add sync at end of conv_ssm_forward (after selective_state_update)
old_end = """\
                is_blackwell=self.is_blackwell,
            )

    def get_state_dtype"""
new_end = """\
                is_blackwell=self.is_blackwell,
            )

        # SM12x Triton workaround: sync after Mamba Triton kernels (vllm#37431)
        if self._needs_triton_sync:
            torch.cuda.synchronize()

    def get_state_dtype"""
assert old_end in content, f'decode end section not found in {path}'
content = content.replace(old_end, new_end)

with open(path, 'w') as f:
    f.write(content)
py_compile.compile(path, doraise=True)
print(f'[OK] Mamba SM12x sync workaround applied to {path}')
PATCH_MAMBA
RUN python3 /tmp/patch_mamba_sync.py

# Final Compilation
RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    # dump git ref in the wheels dir
    git rev-parse HEAD > /workspace/wheels/.vllm-commit

# =========================================================
# STAGE 5: vLLM Wheel Export
# =========================================================
FROM scratch AS vllm-export
COPY --from=vllm-builder /workspace/wheels /

# =========================================================
# STAGE 6: Runner (Installs wheels from host ./wheels/)
# =========================================================
FROM nvcr.io/nvidia/pytorch:26.01-py3 AS runner

# Transferring build settings from build image because of ptxas/jit compilation during vLLM startup
# Build parallemism
ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV VLLM_BASE_DIR=/workspace/vllm

# Set pip cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# Install runtime dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
    curl vim git \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv && pip uninstall -y flash-attn # triton-kernels pytorch-triton

# Set final working directory
WORKDIR $VLLM_BASE_DIR

# Download Tiktoken files
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# Copy wheels from builder stages
COPY --from=flashinfer-builder /workspace/wheels /tmp/wheels/
COPY --from=vllm-builder /workspace/wheels /tmp/wheels/

ARG PRE_TRANSFORMERS=0

# Install all wheels
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    if [ "$PRE_TRANSFORMERS" = "1" ]; then \
        echo "transformers>=5.0.0" > /tmp/tf-override.txt && \
        uv pip install /tmp/wheels/*.whl --override /tmp/tf-override.txt; \
    else \
        uv pip install /tmp/wheels/*.whl; \
    fi && \
    rm -rf /tmp/wheels

# Setup environment for runtime
ARG TORCH_CUDA_ARCH_LIST="12.0a;12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV FLASHINFER_EXTRA_CUDAFLAGS="-DCUTLASS_ENABLE_GDC_FOR_SM100=1"
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings
ENV PATH=$VLLM_BASE_DIR:$PATH


# Final extra deps
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install ray[default] fastsafetensors nvidia-nvshmem-cu13

# Cleanup

# Keeping it here for reference - this won't work as is without squashing layers
# RUN uv pip uninstall absl-py apex argon2-cffi \
#     argon2-cffi-bindings arrow asttokens astunparse async-lru audioread babel beautifulsoup4 \
#     black bleach comm contourpy cycler datasets debugpy decorator defusedxml dllist dm-tree \
#     execnet executing expecttest fastjsonschema fonttools fqdn gast hypothesis \
#     ipykernel ipython ipython_pygments_lexers isoduration isort jedi joblib jupyter-events \
#     jupyter-lsp jupyter_client jupyter_core jupyter_server jupyter_server_terminals jupyterlab \
#     jupyterlab_code_formatter jupyterlab_code_formatter jupyterlab_pygments jupyterlab_server \
#     jupyterlab_tensorboard_pro jupytext kiwisolver matplotlib matplotlib-inline matplotlib-inline \
#     mistune ml_dtypes mock nbclient nbconvert nbformat nest-asyncio notebook notebook_shim \
#     opt_einsum optree outlines_core overrides pandas pandocfilters parso pexpect polygraphy pooch \
#     pyarrow pycocotools pytest-flakefinder pytest-rerunfailures pytest-shard pytest-xdist \
#     scikit-learn scipy Send2Trash soundfile soupsieve soxr spin stack-data \
#     wcwidth webcolors xdoctest Werkzeug