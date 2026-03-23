# Nemotron-3-Super-120B on DGX Spark (SM121) — Optimization Report

> **Period:** 2026-03-18 ~ 2026-03-21 (4 days)
> **Goal:** Maximize inference performance of Nemotron-3-Super-120B-A12B-NVFP4 on NVIDIA DGX Spark
> **Final Results:** Single request average **49.93 tok/s** (ShareGPT), 5 concurrent requests aggregate **63.57 tok/s**

---

## 1. Hardware and Model Overview

### Hardware: NVIDIA DGX Spark (GB10)

| Item | Value |
|------|-------|
| GPU | NVIDIA GB10 (Blackwell) |
| Compute Capability | SM 12.1 (sm_121a) |
| SM Count | 48 (vs 132 on B200) |
| Memory | 128GB LPDDR5X unified memory (CPU+GPU shared) |
| Memory Bandwidth | 241.5 GB/s (sequential) |
| L2 Cache | 24 MB |
| SMEM / SM | 101 KB (vs 228KB on SM100) |
| GPU Clock | 3003 MHz |

### Model: Nemotron-3-Super-120B-A12B-NVFP4 (NemotronH Architecture)

| Item | Value |
|------|-------|
| Total Parameters | 120B |
| Active Parameters | 12B / token |
| Layers | 88 (40 Mamba + 40 MoE + 8 Attention) |
| Layer Pattern | `MEMEMEM*EMEMEMEM*...` |
| MTP Layer Pattern | `*E` (1 Attention + 1 MoE) |
| Routed Experts | 512 |
| Active Experts / token | 22 |
| Shared Experts | 1 (intermediate_size=5376) |
| Quantization | NVFP4 (4-bit floating point) |
| Context Length | 131,072 tokens |
| Mamba SSM cache dtype | float32 |
| `num_nextn_predict_layers` | 1 |

### Software Stack

| Item | Version |
|------|---------|
| vLLM | 0.17.2rc1.dev162 |
| FlashInfer | JIT (CUDA 13.1, compute_121a) |
| CUDA Toolkit | 13.1 |
| PyTorch | 2.10.0a0 |
| Container | Custom build based on nvcr.io/nvidia/nemo:dev |

---

## 2. Complete Attempt History (Chronological)

### ① Initial Attempt with NGC vLLM 26.02 Container

**Attempt:** Run Nemotron-3-Super-120B with `nvcr.io/nvidia/vllm:26.02-py3` (based on vLLM 0.15.1)

**Result:** Immediate failure — vLLM 0.15.1 does not support the Nemotron-3-Super model architecture (NemotronH). Upstream vLLM added ModelOpt mixed precision and NemotronH support in v0.17.0.

> **Reference:** [vLLM v0.17.0 Release](https://github.com/vllm-project/vllm/releases/tag/v0.17.0)

**Action:** Switched to `vllm/vllm-openai:latest` (0.17.1)

---

### ①-b `drop_caches` + `gpu-memory-utilization` Tuning (Unified Memory Issue)

User shared the method of releasing page cache with `sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` → later automated as a `drop-caches` Docker service.

---

### ①-c `gpu-memory-utilization` Tuning (Unified Memory Issue)

**Attempt:** Initially set `gpu-memory-utilization=0.9`

**Problem:** On DGX Spark unified memory, `cudaMemGetInfo` reports only ~44 GiB as free (out of actual 128 GiB, with OS/system using the rest). Requesting 0.9 × 119.7 GiB = 107.73 GiB → crash due to insufficient 44 GiB free.

**Values tried:** 0.9 → 0.45 → 0.7 → 0.8 → 0.85 → final 0.8

**Root fix:** `drop-caches` service to release page cache before startup → freed memory from 41 → 100+ GiB → stable at 0.8

---

### ② NVFP4 GEMM Backend Exploration

Tried each FP4 GEMM path in vLLM one by one:

| Attempt | Backend | Result | Notes |
|---------|---------|--------|-------|
| 1 | `FLASHINFER_CUTLASS` | ❌ `Error Internal` | CUTLASS kernel for SM121 exceeds SMEM |
| 2 | `FLASHINFER_CUDNN` | ❌ `No execution plans support the graph` | cuDNN FP4 execution plan does not support SM121 |
| 3 | `FLASHINFER_CUTEDSL` | ❌ Only `masked_gemm` option exists, SM121 architecture not registered | sm_121 not in CuTe DSL admissible_archs |
| 4 | `VLLM_CUTLASS` | ❌ `Error Internal` (SM120 SMEM overflow) | [vLLM PR #21309](https://github.com/vllm-project/vllm/pull/21309) includes SM120 kernel, but SMEM overflow |
| 5 | **`MARLIN`** | ✅ Works (W4A16 emulation) | **Only one that worked**, requires `VLLM_TEST_FORCE_FP8_MARLIN=1` |

> **Reference:** [FlashInfer #2577](https://github.com/flashinfer-ai/flashinfer/issues/2577) — All mm_fp4 backends fail on SM120
> **Reference:** [vLLM #35566](https://github.com/vllm-project/vllm/issues/35566) — SM120 CUTLASS FP4 MoE garbage output reported
> **Reference:** [SGLang #18954](https://github.com/sgl-project/sglang/issues/18954) — SM120 NVFP4 NaN output reported

---

### ③ Attention Backend Exploration

| Attempt | Backend | Result | Notes |
|---------|---------|--------|-------|
| 1 | `FLASHINFER` (AOT) | ❌ `no kernel image` | AOT build only contains SM90 cubin |
| 2 | `TRITON_ATTN` | ✅ Works | SM121 supported, slow |
| 3 | `FLASH_ATTN` | ❌ SM121 not supported | flash-attn package itself does not support it |
| 4 | **`FLASHINFER` (JIT)** | ✅ Works, fast | JIT compilation with `FLASHINFER_CUDA_ARCH_LIST=12.1a` |

> **Key point:** Need to delete FlashInfer AOT cubin and force JIT compilation for SM121

---

### ③-b NaN Output Debugging

**Symptom:** Model generates tokens (2.5 tok/s) but text is empty. NaN error when requesting logprobs.

**Cause:** NaN/garbage logits produced from the initial FlashInfer CUTLASS FP4 path, causing the detokenizer to return empty strings. Resolved after switching to Marlin.

**Subsequent recurrence:** Empty output recurred every time switching back to FlashInfer CUTLASS FP4 (see ⑪, ⑫, ⑭, ⑮). Format: `reasoning: ""`, `content: null`. 512 tokens generated but all empty strings — FP4 GEMM output is all-zeros or NaN.

---

### ④ First Working Run with Marlin in `--enforce-eager` Mode — **2.5 tok/s**

- Marlin FP4 + TRITON_ATTN + `--enforce-eager` (CUDA graph/torch.compile disabled)
- Model loaded normally, inference works correctly, output is accurate
- But extremely slow at **2.5 tok/s**

---

### ⑤ torch.compile + CUDA graph Enabled — **14.9 tok/s** (6x improvement)

**Attempt:** Remove `--enforce-eager` → torch.compile + CUDA graph (FULL_AND_PIECEWISE)

**Intermediate measurements:**
- First request: 1.9 tok/s (torch.compile JIT overhead, ~710s compilation)
- Second request: **9.5 tok/s** (warming up)
- Steady state: **14.9 tok/s** ✅

**torch.compile cache:** ~710s compilation on first run, saved to `compile_cache`. On subsequent restarts, immediately loaded via `Directly load the compiled graph`.

**Result:** 2.5 → **14.9 tok/s** ✅

---

### ⑤-b Chunked Prefill + Prefix Caching + Memory Profiling Patch Restart — **13.0 tok/s**

> `--enable-chunked-prefill` added (processes long prompts in chunks to reduce memory peak)

**Attempt:** Added `--enable-prefix-caching` + applied memory profiling assertion patch and restarted

**Measurements:**
- **13.0 tok/s** (down from previous 14.9)
- Stable at 12.6~13.0 tok/s

**Reason:** Prefix caching added slight KV cache management overhead, but later recovered to 15.0 tok/s after FlashInfer JIT transition

**Empty content/reasoning output issue:** First confirmed at this point that `reasoning` and `content` in responses were outputting correctly — `"reasoning": "User wants a short greeting in Korean..."`, `"content": "안녕하세요!"`

---

### ⑥ TRITON_ATTN → FlashInfer JIT Transition — **15.0 tok/s**

**Attempt:** Deleted FlashInfer AOT .so files, JIT compiled for sm_121a

**Result:** 14.9 → **15.0 tok/s** (~3% improvement over TRITON)

---

### ⑦ Removing Marlin + Attempting FlashInfer CUTLASS FP4 Transition

**Attempt:** Removed all Marlin environment variables, set `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass` for auto-select

**Issues encountered:**

1. **`scaled_fp4_quant` CUDA sticky error** — The FP4 quantization kernel in vLLM's `_C.abi3.so` asynchronously raises `cudaErrorNoKernelImageForDevice` on SM121, causing all subsequent CUDA operations to fail
   - **Action:** Applied monkey-patch to replace with FlashInfer's `nvfp4_quantize`
   - **Reference:** [vLLM #36821](https://github.com/vllm-project/vllm/issues/36821) — No SM121 aarch64 support

2. **`scaled_fp4_experts_quant`, `silu_and_mul_scaled_fp4_experts_quant`** — Additional C++ kernels in MoE path produce the same sticky error
   - **Action:** Patched these to bypass through FlashInfer as well

3. **`cublasLt.h: No such file or directory`** — FlashInfer `nvfp4_quantize` JIT build requires cuBLAS dev headers
   - **Action:** Added `apt-get install libcublas-dev`

---

### ⑦-b Attempted vLLM Issue Submission (`scaled_fp4_quant` sticky error)

**Attempt:** Searched [existing vLLM issues](https://github.com/vllm-project/vllm/issues?q=is%3Aissue+scaled_fp4_quant+sm121+OR+sm120) for `scaled_fp4_quant` CUDA sticky error and drafted a new issue

**Draft title:** `[Bug]: _C.scaled_fp4_quant produces sticky CUDA error on SM121 (DGX Spark GB10), corrupts CUDA context`

**Related issues found:** [#37141](https://github.com/vllm-project/vllm/issues/37141) (Avarok upstream request), [#36821](https://github.com/vllm-project/vllm/issues/36821) (SM121 unsupported), [#37030](https://github.com/vllm-project/vllm/issues/37030) (Marlin token errors)

**Action:** Collected environment info via `collect_env.py`. Prepared issue submission, but put on hold as the workaround (FlashInfer `nvfp4_quantize` replacement) resolved the problem.

---

### ⑧ Direct Testing of FlashInfer CUTLASS FP4 GEMM (`mm_fp4` API)

**Attempt:** Called `mm_fp4()` API directly inside the container — pure FlashInfer test without going through vLLM

| Test | Result |
|------|--------|
| Small matrix (128x128) | ✅ All 6 tactics succeeded |
| Model-sized (6144x6144) | ✅ Success |
| b.T (transpose) passing | ❌ Shape check failure (`mat2.size(1) == k_packed` — structure where double transpose restores to original) |
| swizzled scale + mm_fp4 | Tested on CUDA context contaminated by previous sticky error — not reproducible |
| raw (no swizzle) + mm_fp4 | ✅ Success |
| Via vLLM call | ❌ `Error Internal` |

**Cause:** During vLLM profiling, when testing various matrix sizes, tiles exceeding SM121's 99KB SMEM (128x128x256B, 256x128x128B) were selected, causing failures. Additionally, the `scaled_fp4_quant` sticky error contaminated the CUDA context, causing cascading failures in subsequent operations.

---

### ⑨ `nemotron_h_mtp` Method Attempt

**Attempt:** `--speculative-config='{"method":"nemotron_h_mtp","num_speculative_tokens":5}'`

**Result:** vLLM automatically switches to `method=mtp`. `nemotron_h_mtp` is remapped to `mtp` internally in vLLM (different name, same code). User asked multiple times "why does it change to mtp?"

> **Reference:** [Nemotron Advanced Deployment Guide](https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.html#vllm) — Includes examples with `num_speculative_tokens > 1`

---

### ⑩ Confirming FlashInfer CUTLASS FP4 Without MTP → Success (MTP Confirmed as OOM Cause)

**Attempt:** Removed MTP, started server with FlashInfer CUTLASS FP4 + PIECEWISE only

**Result:** ✅ Normal startup — confirmed MTP drafter torch.compile as the direct cause of OOM

---

### ⑩-b cudagraph_mode + MTP Memory Estimation Table

Estimated memory consumption for each combination per user request:

| cudagraph_mode | MTP tokens | Model | compile | CUDA graph | KV cache | Total est. | Status |
|---|---|---|---|---|---|---|---|
| NONE | 0 | 70 | 0 | 0 | ~48 | ~118 | ✅ Slow |
| PIECEWISE | 1 | 71 | 6 | 5 | ~33 | ~115 | **✅ Recommended** |
| FULL+PIECE | 1 | 71 | 6 | 13 | ~25 | ~115 | ⚠️ Tight |
| PIECEWISE | 3 | 71 | 10 | 8 | ~24 | ~113 | ⚠️ |
| PIECEWISE | 5 | 71 | 15 | 12 | ~15 | ~113 | ❌ OOM |

---

### ⑩-c MTP 5 Tokens Attempt → OOM

**Attempt:** `num_speculative_tokens=5` (model's `num_nextn_predict_layers=1`)

**Result:** OOM (model 75GiB + MTP drafter torch.compile activation → exceeds 128GiB)

> vLLM warning: *"Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate"*

**Mitigations attempted:**
- `gpu-memory-utilization` 0.9 → 0.85 → 0.8 — insufficient
- `cudagraph_mode=PIECEWISE` — insufficient
- Added 32GiB swap — still OOM (128+32=160GiB fully exhausted, OOM killer triggered)
- `cudagraph_mode=NONE` — Mamba `selective_state_update` assertion error (separate bug)
- `TORCHINDUCTOR_COMPILE_THREADS=1` — only reduces CPU threads, no effect on GPU memory allocation

**Action:** Reduced to MTP 1 token

---

### ⑩ MTP 1 + `mamba_cache_mode=all` → Crash

**Attempt:** MTP + NemotronH (Mamba hybrid) model

**Error:** `selective_state_update` assertion — `state_batch_indices` is None or not 2D

**Cause:** Block-level gather in MTP spec decode metadata incompatible with Mamba2's `mamba_cache_mode=all` state indexing

> **Reference:** eugr/spark-vllm-docker PR #98 mentions the need for `--mamba-cache-mode=align`

**Action:** Switched to `mamba_cache_mode=align` (Phase 6)

---

### ⑪ FlashInfer CUTLASS FP4 + `compute_121a` → Garbage Output, 0.3 tok/s

**Attempt:** After patching all `scaled_fp4_quant` sticky errors, ran the full model with FlashInfer CUTLASS FP4

**Result:** 0.3 tok/s + empty string output (NaN/garbage logits)

> **Reference:** [FlashInfer #2577](https://github.com/flashinfer-ai/flashinfer/issues/2577) — *"CUTLASS returns all zeros on SM120"* reported

---

### ⑫ E2M1 Software Conversion Patch

**Attempt:** Since SM121 lacks the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction, excluded SM121 from the HW E2M1 path in CUTLASS `float_subbyte.h` and `quantization_utils.cuh`, replacing with software conversion. All 3 functions in `quantization_utils.cuh` (`float_to_e2m1`, `e2m1_to_float`, `float_to_e2m1x2`) replaced with SW fallback.

> **Reference:** [RobTand/spark-vllm-docker E2M1 patch](https://raw.githubusercontent.com/RobTand/spark-vllm-docker/flashinfer-pr-patching/e2m1_nvfp4_sm121.patch) — Community patch for the same issue

**Result:** 0.2 tok/s + empty strings — E2M1 was not the problem (slight degradation from 0.3 → 0.2)

**Empty output details:** 512 tokens generated but `reasoning: ""`, `content: null`. All tokens are empty strings — FP4 GEMM output is NaN/all-zeros so all logits have the same value → always same token → detokenizer returns empty string.

**Separate success:** At this point, testing with Marlin backend confirmed **14.7 tok/s** with correct output (Marlin path, not FlashInfer CUTLASS)

---

### ⑬ `fastsafetensors` Loading Format + JIT Prebuild Improvements

**Attempt:** `--load-format=fastsafetensors` (parallel safetensors loader) + changed JIT prebuild to run before model loading (running `prebuild_flashinfer_jit.py` first in entrypoint)

**Result:**
- Model loading time **400s → 92s** (4.3x faster) ✅
- JIT prebuild: ~25 min initially (including fused_moe), subsequent runs skipped via cache marker file
- `MAX_JOBS` limit removed: prebuild uses all CPU cores (20) for parallel build, model loading uses `MAX_JOBS=2` (memory protection)

---

### ⑭ `compute_120f` Attempt (CUDA 13.1 nvcc)

**Attempt:** JIT compiled with TMA fast-path target using `FLASHINFER_CUDA_ARCH_LIST=12.0f`

> **Reference:** [CUTLASS #3096](https://github.com/NVIDIA/cutlass/issues/3096) — 39 tok/s normal output reported with `compute_120f` (SGLang)

**Result:** `cudaErrorMisalignedAddress` → TMA instructions have ABI mismatch with CUDA 12.9 runtime

**Action:** Retry with CUDA 13.1 nvcc + CUDA 13.1 runtime combination

---

### ⑭ `compute_120f` + Full CUDA 13.1 → 4.5 tok/s + Garbage

**Attempt:** `compute_120f` in full CUDA 13.1 environment

**Result:** 4.5 tok/s (better than 0.3) but still empty string output

**Conclusion:** FlashInfer CUTLASS FP4 GEMM does not produce numerically correct results on SM121 (upstream bug)

---

### ⑮ GDC Flag Addition Attempt

**Attempt:** `FLASHINFER_EXTRA_CUDAFLAGS="-DCUTLASS_ENABLE_GDC_FOR_SM100=1"` — compile PDL barriers as actual instructions

> **Reference:** eugr/spark-vllm-docker's GDC flag configuration

**Result:** Still 0.3 tok/s + garbage — GDC alone does not fix it

---

### ⑯ `compute_120f` + SM121 → Illegal Instruction

**Attempt:** Run `FLASHINFER_CUDA_ARCH_LIST=12.0f` on SM121 hardware

**Result:** `cudaErrorIllegalInstruction` — `compute_120f` is SM120 (RTX 5090) exclusive, some instructions incompatible on SM121

> **Reference:** eugr PR #98 — *"FlashInfer compiled with 12.1a, vLLM compiled with 12.0a separately"*

**Action:** Reverted to `12.1a`

---

### ⑰ eugr/spark-vllm-docker Image Attempt

**Attempt:** [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) prebuilt image (CUTLASS 4.4.2, K=64 tiles, GDC flag built-in)

| Attempt | Result |
|---------|--------|
| eugr + FlashInfer CUTLASS FP4 + MTP | ❌ `fp8_gemm_sm100` segfault (`Cannot access data pointer of Tensor that doesn't have storage`) |
| eugr + `kv-cache-dtype=fp8` | ❌ Same segfault |
| eugr + FP8 KV cache disabled | ❌ Same `Cannot access data pointer` on FlashInfer FP4 GEMM |
| eugr + Marlin + FlashInfer attention + PIECEWISE | ❌ FlashInfer TVM FFI segfault |
| eugr + Marlin + `cudagraph_mode=NONE` | ❌ Server starts but crashes on first request with same TVM FFI crash |

> **Cause:** `torch.compile` + TVM FFI incompatibility in eugr image's FlashInfer (main branch build). Did not occur with FlashInfer 0.6.6 pip version in previous custom Dockerfile.

**Action:** Abandoned eugr image, returned to custom Dockerfile

---

### ⑱ TMA WS Grouped GEMM 80 Tactics Skip Problem Analysis

**Symptom:** After successful server startup with custom Dockerfile, autotuner logs show:
```
flashinfer.jit: [Autotuner]: Skipping tactic <MoERunner> 14, due to failure:
[TensorRT-LLM][ERROR] Assertion failed: Failed to initialize cutlass TMA WS grouped gemm.
Error: Error Internal (cutlass_kernel_file_gemm_grouped_sm120_M128_BS_group2.generated.cu:60)
```

**Scale:** Out of all tactics the MoE autotuner tries, **80** TMA Warp Specialized tactics are all skipped, leaving only slow fallback tactics.

**Root cause analysis:**
1. `StageCountAutoCarveout` calculates 3 pipeline stages for SM121's 101KB SMEM
2. In reality, `alignas(1024)` padding + `SharedStorage::TensorStorage` overhead allows only 2 stages
3. Instantiation with 3 stages → SMEM overflow → runtime initialization failure

> **Reference:** [CUTLASS #2820](https://github.com/NVIDIA/cutlass/issues/2820) — SM120 Block-Scaled MMA Runtime Assertion Failure

---

### ⑲ Applying TRT-LLM PR #5823 (C++ `== 100` → `>= 100`)

**Attempt:** [TensorRT-LLM PR #5823](https://github.com/NVIDIA/TensorRT-LLM/pull/5823) — changed `ArchTag::kMinComputeCapability == 100` → `>= 100` on line 390 of `moe_gemm_tma_ws_launcher.inl`

**Result:** SM121 enters the SM100 code path and gets `static assertion failed: Specialization requires Stages set to value 2 or more`

**Cause:** `StageCountAutoCarveout` determines only 1 stage is possible on SM121 (101KB SMEM) — less than half of SM100's 228KB

> **Reference:** [TRT-LLM PR #11997](https://github.com/NVIDIA/TensorRT-LLM/pull/11997), [PR #12309](https://github.com/NVIDIA/TensorRT-LLM/pull/12309) — SM120/SM121 Python-side guard removal

---

### ⑲ Forcing StageCount<2> + Removing SMEM-Overflowing MoE Tiles

**Attempt:** Force `StageCount<2>` instead of `StageCountAutoCarveout` on SM120

**Result:** 128x128x256, 128x256x128 tiles fail with SMEM overflow at `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`

**Action:** Removed SMEM-overflowing tiles from heuristic, dispatcher, and code generator, keeping only 128x128x128

---

### ⑳ Applying CUTLASS commit 73c59c0 (`ReducedSmemCapacityBytes`)

**Attempt:** [CUTLASS commit 73c59c0](https://github.com/NVIDIA/cutlass/commit/73c59c0) — In `sm120_blockscaled_mma_builder.inl`, subtract grouped GEMM overhead (scheduler pipeline, tensor map, CLC response) and calculate stage count using `ReducedSmemCapacityBytes`

**Result:** Still stage < 2 — after subtracting overhead from SM121's 101KB, even the 128x128x128 tile doesn't fit 2 stages

> **Key calculation:** `sm120_smem_capacity_bytes = 101376` (99KB). Subtracting ~10KB grouped GEMM overhead leaves ~91KB. 128x128x128 BlockScaled tile requires ~50KB per stage → 1.8 stages → 1 stage → assertion failure

---

### ㉑ Attempting K=64 Tile (`CtaShape128x128x64B`)

**Attempt:** Added K=64 tile from [FlashInfer PR #2786](https://github.com/flashinfer-ai/flashinfer/pull/2786) to `are_tile_shapes_supported_sm120()`

> **Reference:** [FlashInfer PR #2786](https://github.com/flashinfer-ai/flashinfer/pull/2786) — 2x MoE performance improvement on SM120 with K=64 tiles

**Result:** `undefined symbol: ...tma_warp_specialized_generic_moe_gemm_kernelLauncher<...Shape<128,128,64>...>` — The kernel instantiation code for the K=64 + SM120 + BlockScaled combination is not actually implemented (only declared, no definition)

**Action:** Removed K=64, proceeded with K=128 only

---

### ㉒ Discovery of Unquantized MoE (BF16 TRITON) in MTP Drafter

**Discovery:** Server logs show `Using TRITON backend for Unquantized MoE` warning. User asked: "Why is this appearing?"

**Explanation:** The MTP head of Nemotron-3-Super contains an MoE layer, but unlike the main model, this MoE uses **unquantized BF16 weights**. For BF16 MoE, the Triton kernel is used instead of FlashInfer CUTLASS FP4. This is normal behavior.

---

### ㉒-b RTX PRO 6000 SM120 Benchmark Data Sharing

**User shared:** Detailed benchmark report from 4× RTX PRO 6000 (SM120, 96GB GDDR7):
- Marlin + TP4 + no MTP: **50.5 tok/s** (best)
- Marlin + MTP=2: **39.6 tok/s** (-22% — MTP actually degrades performance with Marlin)
- FlashInfer CUTLASS + MTP=2: **40-41 tok/s** (80 TMA WS tactics skipped)
- **Key finding:** MTP + Marlin causes performance degradation (W4A16 dequantization ≠ native FP4 → acceptance rate 61-85%)

This data provided the rationale for pursuing native FlashInfer CUTLASS FP4 instead of Marlin on DGX Spark.

---

### ㉓ FlashInfer CUTLASS FP4 MoE Success with 128x128x128 Tile Only

**Attempt:** Kept only `CtaShape128x128x64B` (128x128x128 FP4 elements), removed everything else

**Result:**
- ✅ JIT build succeeded (fused_moe_120.so compiled)
- ✅ Autotuner completed normally (no errors, 3 seconds)
- ✅ CUDA graph capture succeeded (51/51)
- ✅ Server started normally

**New problem:** `cudaErrorIllegalInstruction` during inference — crash at Mamba mixer2 (layer 44). Not an MoE issue but an SSM layer problem. The Mamba kernel in vLLM's `_C.abi3.so` is incompatible with SM121.

---

### ㉓ vLLM + FlashInfer Source Build Attempt

**Attempt:** Build vLLM and FlashInfer from source to generate SM121-native `_C.abi3.so`

**Issues encountered:**
- `_C_stable_libtorch` compilation error (vLLM PR #34758, #34302 revert conflict)
- Multiple errors during FlashInfer JIT build (`MAX_JOBS` related, ninja build failures)
- Multiple build failures and retries

> **Reference:** [vLLM PR #34758](https://github.com/vllm-project/vllm/pull/34758) — Build conflict related

**Result:** Eventually succeeded in obtaining SM121-native `_C.abi3.so`. However, the Mamba `cudaErrorIllegalInstruction` issue remained separately (vllm#37431).

---

### ㉔ Triton Mamba Async Bug Fix (`mamba_mixer2_patch.py`)

**Attempt:** Memory barrier omission issue in SM12x Triton PTX codegen reported in vllm#37431

**Symptom:** `cudaErrorIllegalInstruction` / `cudaErrorMisalignedAddress` — async crash in Mamba mixer2 layer

**Fix:** `mamba_mixer2_patch.py` — Insert `torch.cuda.synchronize()` before and after Mamba layers (skip during CUDA graph capture)

> **Reference:** [vLLM #37431](https://github.com/vllm-project/vllm/issues/37431) — SM12x Triton Mamba async bug

**Result:** ✅ Mamba crash resolved

---

### ㉕ Return to Custom Dockerfile + Full Reconfiguration → Success 🎉

**Conclusion:** FlashInfer CUTLASS FP4 GEMM is not production-ready on SM121. The eugr community also uses Marlin for Nemotron.

**Final configuration:**
- **Linear FP4:** FlashInfer CUTLASS (JIT sm_121a)
- **MoE FP4:** FlashInfer CUTLASS (patched heuristic + StageCount<2>)
- **Attention:** FlashInfer (JIT sm_121a)
- **MTP:** Enabled (`mamba_cache_mode=align`)
- **CUDA graph:** `full_and_piecewise`

Achieved stable serving with this combination using custom Dockerfile + 9 patch files

---

### ㉖ Concurrent Request Batching Test — MoE Amortization Discovery

**Attempt:** From ~19.8 tok/s on single requests, checked whether MoE expert reads are amortized within batches

| Concurrent Requests | Aggregate throughput | Drafted/s | Per-user | Multiplier |
|---------------------|---------------------|-----------|----------|------------|
| 1 | 19.8 tok/s | 11.1 | 19.8 | 1.0× |
| 2 | **31.7 tok/s** | 18.0 | 15.9 | **1.60×** |
| 4 | **53.4 tok/s** | 28.4 | 13.4 | **2.69×** |
| 5 | **62.7 tok/s** (server log) | — | 12.5 | **3.17×** |

**Server engine metrics (5 concurrent):**
- Engine generation throughput: **52.7 tok/s** (server log, peak 62.7)
- 4 concurrent MTP counting: Engine **145.8 tok/s** (including thinking tokens), wall-clock per-user **38.3 tok/s**

**Key finding:** MoE expert reads are shared within batches, causing aggregate throughput to scale nearly linearly. The scatter bandwidth bottleneck for single requests is amortized with concurrent requests.

---

### ㉗ Increased to MTP=2 — **~22 tok/s**

**Changes:**
```
--speculative-config={"method":"mtp","num_speculative_tokens":2}  # 1 → 2
--max-num-batched-tokens=8400  # 8352 → 8400 (MTP=2 block size requirement)
```

**Issue encountered:** `AssertionError: block_size (8400) must be <= max_num_batched_tokens (8352)` → resolved with `max_num_batched_tokens=8400`

**Detailed measurements (MTP=1 vs MTP=2):**

| Test Type | MTP=1 | MTP=2 | Improvement |
|-----------|-------|-------|-------------|
| Counting (repetitive, high acceptance) | 21.8 tok/s | **27.0 tok/s** | **+24%** |
| Essay (creative) | 19-20 tok/s | 20.7 tok/s | +5% |
| Code generation | 19 tok/s | ~25 tok/s | +32% |

**MTP=2 server metrics:**
- Mean acceptance length: 2.33~2.84 (max 3.0)
- Per-position acceptance rate: [79-97%, 53-88%]
- Forward pass rate (drafted throughput): **~11.2 fwd/s** (90ms per forward, limited by MoE scatter bandwidth)
- Drafted token throughput: ~18 tok/s (with MTP=2: 11.2 × (1+accepted) ≈ 18)

**CUDA_LAUNCH_BLOCKING=1 test:** When tried as Mamba Triton async bug workaround: 14→8.8 tok/s (**37% performance degradation**). The `torch.cuda.synchronize()` workaround is more efficient.

**Result:** 19.8 → **~22 tok/s** (+12%) — MTP acceptance rate [79-97%, 53-88%]

---

### ㉔ Transition from Marlin to Without Marlin

In fact, the final configuration **does not use Marlin**. In previous attempts:
- Marlin + FlashInfer Attention: **14.6 tok/s** (including Marlin MoE)
- But MTP acceptance rate was low with Marlin (W4A16 dequantization differs numerically from native FP4)

> **Reference:** RTX PRO 6000 SM120 benchmark showed Marlin + MTP = **-22% performance degradation** (acceptance rate 61-85%)

Ultimately patched FlashInfer CUTLASS FP4 to transition MoE to native FP4 as well → numerical consistency with MTP → high acceptance rate

---

## 3. Attempted Optimizations That Failed or Had No Effect (Final Session)

### ❌ FlashInfer 0.6.6 pip Upgrade (Attempted FlashInfer CUTLASS FP4 Fix)

**Attempt:** Upgraded FlashInfer from 0.6.5 → 0.6.6. Expected [FlashInfer PR #2670](https://github.com/flashinfer-ai/flashinfer/pull/2670) (SM120 SMEM fix, `StageCount<2>`) to be included.

**Result:** Same `Error Internal` — PR #2670's fix was either not included in pip 0.6.6 release or not applied to the MoE grouped GEMM path. Only SM121 recognition was added ([PR #2631](https://github.com/flashinfer-ai/flashinfer/pull/2631)).

---

### ❌ swizzle_blockscale / `is_sf_swizzled` Test

**Attempt:** Comparison test of `scaled_fp4_quant(x, gsf, True)` (swizzled=True) vs `scaled_fp4_quant(x, gsf, False)` (raw)

**Result:** Both produced CUDA sticky errors in subsequent operations. Regardless of swizzle setting, the FP4 quantization kernel in vLLM's `_C.abi3.so` itself produces sticky CUDA errors on SM121.

---

### ❌ `flashinfer-cudnn` Backend (cuBLASLt FP4)

**Attempt:** cuDNN/cuBLASLt FP4 GEMM to bypass CUTLASS SMEM issues

**Error:** `cudnn._compiled_module.cudnnGraphNotSupportedError: No execution plans support the graph`

**Cause:** cuDNN's FP4 execution plan does not support SM121

---

### ❌ `compile_sizes=[]` (Disable torch.compile Size Specification)

**Attempt:** Skip warmup compilation with `compile_sizes=[]` to avoid MTP drafter's torch.compile OOM

**Result:** Had memory savings effect, but separately failed due to Mamba `selective_state_update` assertion error in `cudagraph_mode=NONE`

---

### ❌ BF16 → FP16 Activation Casting / FP32 Scale Precision

**Attempt:**
- Applied the fix from `flashinfer#2577` report: "converting scale factor to `.float()` resolves the issue"
- Converted BF16 activations to FP16 before passing to CUTLASS FP4 GEMM

**Result:** FlashInfer CUTLASS FP4 GEMM output still garbage (0.3 tok/s, empty strings)

---

### ❌ vLLM Memory Profiling Patches (2 cases)

**Attempt:** Due to `cudaMemGetInfo` reporting inaccurate free memory on DGX Spark unified memory:
- `vllm/utils.py` startup memory check patch (bypass free < requested check)
- `vllm/worker/gpu_worker.py` profiling assertion patch (bypass `init_free < post_profile_free` check)

**Result:** Worked, but later root-fixed with `drop-caches` service, making patches unnecessary

---

### ❌ causal-conv1d / mamba-ssm Native CUDA Kernel Attempt

**Attempt:** Replace with [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) and [mamba-ssm](https://github.com/state-spaces/mamba) native CUDA kernels to bypass Triton Mamba kernel's SM121 async bug

> **Reference:** [vLLM #37431](https://github.com/vllm-project/vllm/issues/37431) — User shared asking "isn't this the same issue?"

**Result:** Build failed due to extension compatibility issues with NGC 26.01 PyTorch. Applied `torch.cuda.synchronize()` workaround instead (`mamba_mixer2_patch.py`)

---

### ❌ NVIDIA Driver 595.71 Instability / System Reboot

**Symptom:** Multiple OOMs, swap exhausted, then `nvidia-smi` fails to detect GPU

**Attempt:** Checked `nvidia-smi` → driver state abnormal → system reboot

**Result:** Normal recovery after reboot. Stabilized afterwards with `drop-caches` service + `gpu-memory-utilization=0.8`

---

### ❌ `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` (Block FP8 ScaledMM)

**Attempt:** FlashInfer FP8 ScaledMM path causes segfault on SM121, blocked with environment variable

**Result:** Worked. But later blocked at code level in `patch_fp8_sm120.py` for SM120+, making the environment variable unnecessary.

---

### ❌ `restart: unless-stopped` Setting

**Attempt:** Docker restart policy for automatic restart on server crash

**Result:** Risk of infinite loop in unified memory environment: OOM crash → restart → re-OOM. Final decision: `restart: none`

---

### ❌ `VLLM_COMPILE` Environment Variable (torch.compile mode)

**Attempt:** Enable torch.compile with `VLLM_COMPILE=1` environment variable (initial approach)

**Result:** Later switched to `--compilation-config` CLI argument for finer control. `VLLM_COMPILE` environment variable removed.

---

### ❌ `VLLM_SKIP_GPU_MEM_CHECK=1` (Memory Check Bypass)

**Attempt:** Bypass startup memory check with environment variable due to inaccurate `cudaMemGetInfo` free memory on DGX Spark unified memory

**Result:** Worked, but releasing page cache with `drop-caches` service reports accurate free memory → environment variable unnecessary. Removed along with `utils.py` + `gpu_worker.py` memory patches in Dockerfile.

---

### ❌ `torch.set_float32_matmul_precision('high')` Warning

**Symptom:** Warning during torch.compile: `TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled`

**Attempt:** Considered applying the setting

**Result:** No effect on FP4 models as float32 matmul is almost nonexistent. Just ignored the warning.

---

### Other Docker Settings (Not Performance-Related, For Stability)

| Setting | Value | Reason |
|---------|-------|--------|
| `ipc: host` | Host IPC sharing | CUDA IPC, NCCL communication |
| `shm_size: "16g"` | 16GB shared memory | PyTorch DataLoader, NCCL buffers |
| `ulimits: memlock: -1` | Memory lock limit removed | GPU memory pinning |
| `--enable-sleep-mode` | Release GPU memory when idle | Unified memory savings |
| `--tool-call-parser=qwen3_coder` | Tool call parser | Nemotron-3-Super compatible |
| `--reasoning-parser=super_v3` | Reasoning parser | [super_v3_reasoning_parser.py](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) |
| `docker compose build` | Custom Dockerfile build | Build patched image (~6 min) |

---

### ❌ `fuse_attn_quant: true` — Attention-Quantization Fusion

**Attempt:**
```
"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true,"fuse_attn_quant":true}
```

**Error:** `AssertionError: Query must be FP8 when attn+quant fusion happened`

**Cause:** `fuse_attn_quant` expects FP8 query, but `disable_flashinfer_q_quantization: true` is required for SM120 XQA compatibility. BF16 query + fuse_attn_quant is fundamentally incompatible.

**Side effect:** CUDA graph mode downgraded from `FULL_AND_PIECEWISE` → `FULL_DECODE_ONLY`

**Action:** Removed `fuse_attn_quant`

---

### ❌ `max_autotune: true` — GEMM Auto-Tuning

**Attempt:** `"pass_config":{"max_autotune":true,...}`

**Error:** `Not enough SMs to use max_autotune_gemm mode`

**Cause:** GB10's 48 SMs are below the GEMM autotuning threshold
**Side effect:** Compilation time increased (55→91s), Mamba cache blocks decreased (147→117)

**Action:** Immediately reverted

---

### ❌ THP (Transparent Huge Pages) Activation

**Attempt:** Enable THP + defrag in privileged container (using docker service since sudo unavailable)
```sh
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo always > /sys/kernel/mm/transparent_hugepage/defrag
```

**Result:** No measurable effect on GPU performance. THP is meaningless in unified memory architecture.

---

### ❌ MTP=3 — 3 Speculative Tokens

**Attempt:**
```
--speculative-config={"method":"mtp","num_speculative_tokens":3}
--max-num-batched-tokens=8448
```

**Issue encountered:** `AssertionError: block_size (8432) must be <= max_num_batched_tokens (8400)` → resolved with 8448

**Result:** Same or worse compared to MTP=2
- 3rd position acceptance rate: 48-59% (too low, overhead > benefit)
- This model has `num_nextn_predict_layers=1` so the same MTP layer repeats 3 times → increasingly inaccurate

**Action:** Reverted to MTP=2

---

### ❌ MTP=5 — 5 Speculative Tokens

**Attempt (initial session):** `num_speculative_tokens=5`

**Result:** OOM — model 75GiB + MTP drafter 5x forward activation → exceeds 128GiB unified memory

**Mitigations attempted:**
- Added 32GiB swap → still OOM (128+32=160GiB fully exhausted)
- `gpu-memory-utilization` 0.9→0.85→0.8 → insufficient
- `TORCHINDUCTOR_COMPILE_THREADS=1` → only reduces CPU, no GPU memory effect

**Action:** Reduced to MTP=1, later settled on MTP=2

---

### ❌ nsys GPU Profiling (4 attempts)

**Attempt 1:** `nsys profile ... python3 -m vllm.entrypoints.openai.api_server`
- Result: "SKIPPED: does not contain CUDA kernel data" — CUDA graph replay is not decomposed

**Attempt 2:** Added `--cuda-graph-trace=node` + 540s delay (waiting for server loading)
- Result: 2079 CUPTI events collected. `"Hardware tracing used for CUDA tracing"` — kernel activity table not generated
- Judged to be a limitation of SM121 hardware tracer

**Attempt 3:** `--cuda-trace-scope=system-wide` (from docker exec)
- Result: No CUDA trace data collected due to container isolation

**Attempt 4:** `torch.profiler` (after CUDA context initialization in client process)
- Result: Only captured `cudaDeviceSynchronize` 650us — server GPU kernels not visible (CUPTI is per-process)

**Attempt 5:** `nsys profile --attach-process=157` (EngineCore PID)
- Result: `unrecognised option` — option not available in nsys 2025.6.1

**Conclusion:** nsys GPU kernel profiling currently not possible on SM121

---

### ❌ `--kv-cache-dtype=fp8` (eugr recipe)

**Attempt:** FP8 KV cache in eugr `vllm-node` image

**Result:** `Cannot access data pointer of Tensor that doesn't have storage` — FlashInfer `bmm_fp8` → `fp8_gemm_sm100` segfault

**Cause:** FlashInfer FP8 ScaledMM is unstable on SM121

---

### ❌ `cudagraph_mode=NONE` + MTP → Mamba Assertion

**Error:** `selective_state_update` assertion — MTP + NemotronH combination bug regardless of cudagraph_mode

**Action:** Resolved with `mamba_cache_mode=align` (Phase 6)

---

## 4. Performance Progression

| Phase | Commit | Key Change | tok/s (single) | Change |
|-------|--------|------------|----------------|--------|
| Initial attempt | — | NGC 26.02 (vLLM 0.15.1) | Failed | — |
| Marlin + enforce-eager | — | TRITON_ATTN, no compile | 2.5 | — |
| +torch.compile (first warmup) | — | JIT overhead | 1.9 → 9.5 | Warming up |
| +torch.compile + CUDA graph | — | FULL_AND_PIECEWISE | **14.9** | **+496%** |
| +prefix caching | — | KV cache reuse | 13.0 | -13% (overhead) |
| +FlashInfer JIT Attention | — | TRITON → FLASHINFER | **15.0** | +15% |
| FlashInfer CUTLASS FP4 attempt | — | compute_121a | 0.3 | garbage output |
| +E2M1 patch | — | SW E2M1 conversion | 0.2 | still garbage |
| +compute_120f (CUDA 13.1) | — | TMA fast-path | 4.5 | still garbage |
| Marlin MoE (standalone) | — | All Marlin | 14.6 | correct output |
| +E2M1 patch then Marlin test | — | Marlin + FlashInfer Attn | 14.7 | correct output |
| 0 | `5b1fb7e` | Initial setup (custom Dockerfile) | Failed | — |
| 1 | `c242f9b` | FlashInfer CUTLASS FP4 + MTP attempt | 0.3 | garbage |
| 2 | `8ad7a15` | SM121 runtime patches (3 types) | **~12.0** | **Baseline** |
| — | — | (first warmup 1.2, then 12.3~12.4) | 12.3 | — |
| 3 | `dd3a3e3` | torch.compile (no CUDA graph) | ~12.4 | +3% |
| — | — | +fuse_norm_quant (peak 12.8, stable 12.6) | 12.6 | — |
| 4 | `3b561e7` | fuse_norm_quant + fuse_act_quant | ~12.8 | +3% |
| 5 | `d70b5bf` | MTP speculative (tokens=1) | **~17.1** | **+33%** ⭐ |
| — | — | (MTP 100 tok 5.8s=17.2, 200 tok 13s=15.4) | 15.4~17.1 | — |
| 6 | `70b814c` | Prefix caching + mamba_cache_mode=align | ~17.1 | — |
| — | — | (peak 15.8, average 17.1) | 15.8~17.1 | — |
| 7 | `be637ce` | CUDA graph PIECEWISE | ~17.9 | +5% |
| 8 | `a412be1` | SM121 CUDA graph patches (3 types) (FULL_AND_PIECEWISE) | **~19.8** | +11% |
| — | — | (peak 21.8 counting, 19.5 general) | 19.5~21.8 | — |
| — | — | +CUDA_LAUNCH_BLOCKING=1 (Mamba workaround) | 8.8 | -56% (not adopted) |
| 9 | uncommitted | MTP=2 (counting 27.0, essay 20.7) | **~22.1** | +12% |
| 10a | uncommitted | CUTLASS alignas(1024)→alignas(128) | 39.9 (ShareGPT) | **+80%** ⭐ |
| 10b | uncommitted | + StageCount<4> forced | **48.9** (ShareGPT) | **+22%** ⭐ |
| — | — | StageCount<3> attempt | 46.15 | -6% (not adopted) |
| — | — | StageCount<2> attempt | 40.05 / 71.00 | -18% (not adopted) |
| — | — | StageCount<5> attempt | segfault | — |
| — | — | 128×128×128B tile | segfault | — |
| — | — | 256×128×64B tile | segfault | epilogue 64KB > SMEM |
| — | — | --use_fast_math | 27.2 | -44% (not adopted) |
| — | — | SchedulerPipelineStageCount=2 | 40.76 | -17% (not adopted) |
| — | — | MTP=3 | 39.84 / 48.82 | -18% (not adopted) |
| — | — | MTP=1 + SC4 | 47.56 / 74.88 | concurrent only better |
| — | — | BTankut GB10 Triton config (K=256) | 39.96 | -18% (unsuitable for BF16) |
| — | — | **Total improvement over Baseline (12.0)** | — | **+308%** |
| — | — | **Total improvement over initial (2.5)** | — | **+1857%** |

---

## 5. Benchmark Results

> Benchmark run date: 2026-03-22
> vLLM version: 0.17.2rc1.dev162
> Benchmark tool: `vllm bench serve` (official CLI, [docs](https://docs.vllm.ai/en/latest/benchmarking/cli/#online-benchmark))
> Dataset: `--dataset-name random --input-len 128 --output-len 256`
> Measurement method: rate=1 (10 prompts, single user simulation) + rate=inf (20 prompts, max throughput)

### 5-0. Benchmarks by Configuration Combination (12 combinations)

Each configuration combination was measured by restarting the server and running `vllm bench serve`.

| # | Configuration | Attention | Out tok/s (rate=1) | Out tok/s (rate=inf) | TPOT (ms) | P99 ITL (ms) | MTP Accept |
|---|---------------|-----------|-------------------:|---------------------:|----------:|-------------:|-----------:|
| 1 | **MTP=2 + CUDA-graph(f&p) + fusion** | FLASHINFER | 32.86 | 37.94 | 146.53 | — | 19.93% |
| 2 | MTP=1 + CUDA-graph(f&p) + fusion | FLASHINFER | 47.16 | 56.07 | 151.58 | — | 35.76% |
| 3 | MTP=0 + CUDA-graph(f&p) + fusion | FLASHINFER | 47.94 | 89.25 | 176.68 | — | — |
| 4 | MTP=2 + CUDA-graph(piece) + fusion | FLASHINFER | 30.63 | 36.14 | 158.83 | — | 14.68% |
| 5 | MTP=2 + no-CUDA-graph + fusion | FLASHINFER | 43.18 | 72.89 | 165.04 | — | 19.55% |
| 6 | MTP=2 + CUDA-graph(f&p) + no-fusion | FLASHINFER | 34.19 | 44.42 | 154.47 | 381.11 | 16.96% |
| 7 | MTP=0 + enforce-eager | FLASHINFER | 54.54 | 93.52 | 151.60 | 310.82 | — |
| 8 | MTP=1 + CUDA-graph(piece) + fusion | FLASHINFER | 53.75 | 101.66 | 151.36 | 312.89 | — |
| 9 | MTP=0 + CUDA-graph(piece) + fusion | FLASHINFER | 51.95 | 96.13 | 160.59 | — | — |
| 10 | MTP=0 + torch.compile-only + fusion | FLASHINFER | 52.84 | 96.31 | 158.07 | — | — |
| 11 | MTP=2 + CUDA-graph(f&p) + fusion | **TRITON_ATTN** | 38.76 | 39.16 | 140.12 | 376.79 | — |
| 12 | MTP=0 + enforce-eager | **TRITON_ATTN** | 53.01 | 91.70 | 156.71 | 313.06 | — |

**Key analysis:**

- **MTP is counterproductive on random datasets**: With acceptance rate of 15-36%, there are no predictable patterns, so MTP overhead exceeds its benefits. With real text, MTP=2 is optimal with 66-92% acceptance (see 5-1 below).
- **CUDA graph effect**: With MTP=0: piecewise(51.95) vs none(52.84) vs enforce-eager(54.54) — not much difference on random datasets. However, CUDA graphs reduce latency jitter in real usage.
- **Fusion effect**: With MTP=2: fusion ON(32.86) vs OFF(34.19) — minimal on random, but 8ms TPOT difference (146→154ms).
- **Attention**: FLASHINFER(32.86) vs TRITON_ATTN(38.76) — TRITON slightly faster on random, but FlashInfer shows better MTP acceptance on real text.
- **Max throughput (rate=inf)**: MTP=1+piecewise is highest at **101.66 tok/s** (random dataset basis).

---

### 5-1. Real Text — Single Request (Single User Experience)

5 different programming/technical prompts, `max_tokens=512`, `temperature=0.7`

| Request | Output Tokens | Duration | Throughput |
|---------|--------------|----------|------------|
| Python quicksort implementation | 512 | 21.7s | **23.6 tok/s** |
| Relativity theory explanation | 512 | 26.2s | **19.6 tok/s** |
| Docker guide | 512 | 24.9s | **20.5 tok/s** |
| BST implementation | 512 | 20.6s | **24.8 tok/s** |
| Computing history | 512 | 23.2s | **22.0 tok/s** |

| Metric | Value |
|--------|-------|
| **Average Throughput** | **22.1 tok/s** |
| Minimum | 19.6 tok/s |
| Maximum | 24.8 tok/s |

**Server log MTP metrics (single request):**
- Mean acceptance length: 2.33~2.84
- Per-position acceptance rate: [79-97%, 53-88%]
- Draft acceptance rate: 66-92%

### 5-2. Real Text — 5 Concurrent Requests (Multi-User Throughput)

5 different prompts sent simultaneously, `max_tokens=512`

| Request | Output Tokens | Duration | Per-User Throughput |
|---------|--------------|----------|---------------------|
| Web scraper implementation | 512 | 49.2s | 10.4 tok/s |
| Quantum computing explanation | 512 | 49.2s | 10.4 tok/s |
| FastAPI REST API | 512 | 46.1s | 11.1 tok/s |
| ML pipeline explanation | 512 | 51.5s | 9.9 tok/s |
| SQL vs NoSQL comparison | 512 | 50.7s | 10.1 tok/s |

| Metric | Value |
|--------|-------|
| **Aggregate Throughput** | **49.7 tok/s** |
| **Average Per-User** | **10.4 tok/s** |

### 5-3. vllm bench serve — Random Dataset

`--dataset-name random --input-len 128 --output-len 256`

#### Request Rate = 1.0 (20 prompts)

| Metric | Value |
|--------|-------|
| Output token throughput | 38.05 tok/s |
| Peak output throughput | 42.00 tok/s |
| Total token throughput | 57.07 tok/s |
| Mean TPOT | 154.88 ms |
| Mean ITL | 206.78 ms |
| P99 ITL | 374.14 ms |
| MTP Acceptance rate | 16.88% |

#### Request Rate = inf (50 prompts, max throughput)

| Metric | Value |
|--------|-------|
| Output token throughput | **42.57 tok/s** |
| Total token throughput | **63.86 tok/s** |
| Mean TPOT | 155.23 ms |
| P99 ITL | 371.83 ms |
| MTP Acceptance rate | 18.65% |

> **Note:** MTP acceptance ~17% on random datasets is due to lack of predictable patterns. In real usage: 66-92%.

---

## 6. Patch File List

### Runtime Patches (Volume Mount)

| File | Lines | Role | Mount Target |
|------|-------|------|-------------|
| `tvm_build_patch.py` | 918 | TVM FFI C DLPack `has_storage()` fix (related to [PyTorch #122706](https://github.com/pytorch/pytorch/issues/122706)) | `tvm_ffi/utils/_build_optional_torch_c_dlpack.py` |
| `cutlass_heuristic_patch.cpp` | 769 | Restrict tile sizes to fit SM121 SMEM (`CtaShape128x128x64B` only) | `flashinfer/data/.../cutlass_heuristic.cpp` |
| `prebuild_flashinfer_jit.py` | 107 | Delete AOT `fused_moe_120.so` + JIT prebuild + cache marker | `/workspace/prebuild_flashinfer_jit.py` |
| `vllm_flashinfer_utils_patch.py` | 796 | `supports_trtllm_attention()` SM120 support + `UNIFORM_BATCH` CG support return (previously only returning `UNIFORM_SINGLE_TOKEN_DECODE`, preventing FULL graph) | `vllm/utils/flashinfer.py` |
| `flashinfer_backend_patch.py` | 1,818 | `get_required_kv_cache_layout()` SM120 HND return + causal attention mask generation when `q_seq_len > 1` during XQA spec decode | `vllm/v1/attention/backends/flashinfer.py` |
| `mamba_mixer2_patch.py` | 956 | `torch.cuda.synchronize()` before/after `conv_ssm_forward` (SM121 Triton PTX codegen memory barrier omission workaround, [vllm#37431](https://github.com/vllm-project/vllm/issues/37431)), skipped during CUDA graph capture via `is_current_stream_capturing()` check | `vllm/.../mamba/mamba_mixer2.py` |

### Build-Time Patches (Dockerfile)

| File/Patch | Role |
|------------|------|
| `patch_sm121_moe.py` | Add `sm_120a`/`sm_121a` to CuTe DSL `admissible_archs` (18 locations), TRT-LLM launcher `ICHECK_EQ(major, 10)` → `ICHECK_GE(major, 10)` |
| `patch_fp8_sm120.py` | Block `FlashInferFP8ScaledMMLinearKernel` on SM120+ (prevent segfault) |
| `fix_quantization_utils_sm121.py` | `cvt.rn.satfinite.e2m1x2.f32` PTX unsupported → software E2M1 conversion |
| `scaled_fp4_quant` patch | Bypass SM121 sticky error in vLLM `_C.abi3.so` → replace with FlashInfer `nvfp4_quantize` |
| PR #5823 patch | `moe_gemm_tma_ws_launcher.inl` line 390: `== 100` → `>= 100` |
| CUTLASS 73c59c0 patch | `sm120_blockscaled_mma_builder.inl`: `ReducedSmemCapacityBytes` reflecting grouped GEMM overhead |
| StageCount<2> patch | Force 2 pipeline stages in SM120 MoE grouped GEMM |
| generate_kernels.py patch | Remove SMEM-overflowing tiles (128x128x256, 128x256x128, 256x128x128) |

---

## 7. External References (Issues, PRs, Documents)

### vLLM
| Issue/PR | Title | Relevance |
|----------|-------|-----------|
| [#21309](https://github.com/vllm-project/vllm/pull/21309) | Add CUTLASS NVFP4 SM120 kernels | SM120 FP4 GEMM addition (SMEM overflow unresolved) |
| [#32093](https://github.com/vllm-project/vllm/issues/32093) | FlashInfer SM120 support related | Shared by user for reference |
| [#33726](https://github.com/vllm-project/vllm/pull/33726) | Mamba2 + spec decode support | Introduced `mamba_cache_mode=align` |
| [#34758](https://github.com/vllm-project/vllm/pull/34758) | vLLM main build related revert | Conflict during eugr source build |
| [#35566](https://github.com/vllm-project/vllm/issues/35566) | SM120 CUTLASS FP4 MoE garbage output | `compute_120f` required reported |
| [#36094](https://github.com/vllm-project/vllm/issues/36094) | Qwen3.5 NVFP4 accuracy degradation | Also occurs on SM100 — possible model checkpoint issue |
| [#36821](https://github.com/vllm-project/vllm/issues/36821) | No sm_121 support on aarch64 — DGX Spark | SM121 aarch64 unsupported |
| [#36865](https://github.com/vllm-project/vllm/issues/36865) | SM120/RTX 5090 source build unsupported targets | Source build architecture issues |
| [#37030](https://github.com/vllm-project/vllm/issues/37030) | MXFP4 on SM121 - Marlin kernel wrong tokens | SM121 Marlin token errors |
| [#37141](https://github.com/vllm-project/vllm/issues/37141) | Upstream DGX Spark improvements from Avarok/dgx-vllm | DGX Spark patch upstream request |
| [#33333](https://github.com/vllm-project/vllm/issues/33333) | FLASHINFER_CUTLASS not supported on SM120 | FlashInfer CUTLASS FP4 SM120 unsupported |
| [#33416](https://github.com/vllm-project/vllm/issues/33416) | NVFP4 MoE kernels fail on RTX Blackwell SM12.0 | MoE kernel failure report |
| [#36453](https://github.com/vllm-project/vllm/pull/36453) | SM120 patch PR | Added `is_device_capability_family(120)` |
| [#37431](https://github.com/vllm-project/vllm/issues/37431) | Triton Mamba async bug on SM12x | `torch.cuda.synchronize()` workaround needed — same issue as our mamba_mixer2_patch.py |
| [#21274](https://github.com/vllm-project/vllm/issues/21274) | nvfp4 support on sm120 | Closed (Not planned) — initial SM120 FP4 support request |

### FlashInfer
| Issue/PR | Title | Relevance |
|----------|-------|-----------|
| [#2252](https://github.com/flashinfer-ai/flashinfer/issues/2252) | FlashInfer SM120 support | Initially shared by user for reference |
| [#2577](https://github.com/flashinfer-ai/flashinfer/issues/2577) | SM120 mm_fp4 all backends fail | CUTLASS all zeros, cuDNN unsupported. Report that converting scale to `.float()` resolves |
| [#2631](https://github.com/flashinfer-ai/flashinfer/pull/2631) | SM121 recognition added | SM121 recognition included in FlashInfer 0.6.6, but FP4 GEMM execution still fails |
| [#2670](https://github.com/flashinfer-ai/flashinfer/pull/2670) | SM120 FP4 shared memory fix (StageCount) | tinygemm2 SMEM allocation reduction (FlashInfer 0.6.6) |
| [#2716](https://github.com/flashinfer-ai/flashinfer/pull/2716) | GDC barrier fix | PDL sync fix |
| [#2725](https://github.com/flashinfer-ai/flashinfer/pull/2725) | SM120 MoE backend selection patch PR | Added `is_device_capability_family(120)` in 5 files |
| [#2786](https://github.com/flashinfer-ai/flashinfer/pull/2786) | Add K=64 tiles for SM120 MoE | 2x MoE performance improvement with K=64 tiles |
| [#2835](https://github.com/flashinfer-ai/flashinfer/pull/2835) | SM120 SMEM arch detection (Python/CuTe DSL) | `get_blackwell_smem_arch()` to detect SM120 vs SM100 — same issue as our manual patch |

### CUTLASS
| Issue/PR/Commit | Title | Relevance |
|-----------------|-------|-----------|
| [#2185](https://github.com/NVIDIA/cutlass/pull/2185) | SM120 GEMM fix related | Shared by user for reference (checking inclusion in v0.6.6) |
| [#2800](https://github.com/NVIDIA/cutlass/issues/2800) | BlockScaledMmaOp restricts FP4 to sm_100a only | Root cause of FP4 MMA unavailability on SM120 |
| [#2820](https://github.com/NVIDIA/cutlass/issues/2820) | SM120 Block-Scaled MMA Runtime Assertion Failure | StageCount assertion failure root cause analysis |
| [#3096](https://github.com/NVIDIA/cutlass/issues/3096) | SM120 FP4 GEMM performance | 39 tok/s with `compute_120f`, 14.6 tok/s with `compute_120a`. 10+ patches documented |
| [73c59c0](https://github.com/NVIDIA/cutlass/commit/73c59c055c0fec87792470dbf33325158113db5e) | ReducedSmemCapacityBytes for grouped GEMM | Reflecting scheduler pipeline/tensor map overhead in SM120 grouped GEMM |

### TensorRT-LLM
| Issue/PR | Title | Relevance |
|----------|-------|-----------|
| [#5823](https://github.com/NVIDIA/TensorRT-LLM/pull/5823) | Fix MoE regression for SM120 | `== 100` → `>= 100` (C++ level). merged but NOT propagated to FlashInfer |
| [#11368](https://github.com/NVIDIA/TensorRT-LLM/issues/11368) | FP4 CUTLASS GEMM fails on SM121 (shared memory overflow) | 128x128x256 tile exceeds SM121 SMEM |
| [#11997](https://github.com/NVIDIA/TensorRT-LLM/pull/11997) | SM120/SM121 Python guard removal | `{100, 103}` → `{100, 103, 120, 121}` |
| [#12309](https://github.com/NVIDIA/TensorRT-LLM/pull/12309) | SM120 MoE backend support | `NotImplementedError` guard removal |

### SGLang
| Issue | Title | Relevance |
|-------|-------|-----------|
| [#18954](https://github.com/sgl-project/sglang/issues/18954) | SM120 NVFP4 NaN output | Both FlashInfer CUTLASS/cuDNN produce NaN. FP8 works with Triton GEMM |
| [#19637](https://github.com/sgl-project/sglang/issues/19637) | SM120 Performance Optimization Plan | Performance optimization roadmap |
| [#20050](https://github.com/sgl-project/sglang/issues/20050) | TP8/TP4 gibberish, TP2 normal | TP configuration issue (unrelated to SM120) |

### NVIDIA Official Documentation/Forums
| Link | Content |
|------|---------|
| [Nemotron-3-Super Advanced Deployment Guide](https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.html#vllm) | MTP `num_speculative_tokens > 1` usage example. User referenced: "they use values greater than 1 here" |
| [NVIDIA Forum: 50% improvement on Spark](https://forums.developer.nvidia.com/t/50-improvement-on-spark/363493) | DGX Spark 50% performance improvement case |
| [NVIDIA Forum: Run vLLM in Spark (page 4)](https://forums.developer.nvidia.com/t/run-vllm-in-spark/348862/61?page=4) | DGX Spark vLLM community thread. `12.1f` suggestion |
| [NVIDIA Forum: Custom FP4 CUDA kernel 129 TFLOPS on DGX Spark](https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600) | Custom FP4 kernel achieving 129 TFLOPS |
| [NVIDIA Forum: "When are you going to fix tcgen05 FP4 for SM121"](https://forums.developer.nvidia.com/t/dearest-cutlass-team-when-the-hell-are-you-going-to-properly-fix-tcgen05-fp4-support-for-dgx-spark-gb10-sm121/359598) | Community frustration about lack of SM121 FP4 support |
| [NVIDIA Forum: SM121 software support lacking](https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663) | SM121 software support roadmap request |
| [NVIDIA Forum: vLLM 0.17.0 MXFP4 patches for DGX Spark](https://forums.developer.nvidia.com/t/vllm-0-17-0-mxfp4-patches-for-dgx-spark-qwen3-5-35b-a3b-70-tok-s-gpt-oss-120b-80-tok-s-tp-2/362824) | Qwen3.5-35B 70 tok/s, GPT-OSS-120B 80 tok/s achievement |
| [NVIDIA Forum: SM121 CUTLASS kernel optimization](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960) | NVFP4 356 TFLOPS MoE grouped GEMM on DGX Spark. `sm_121a` → `BlockScaledMmaOp.admissible_archs` addition |
| [vLLM Benchmarking CLI Docs](https://docs.vllm.ai/en/latest/benchmarking/cli/#online-benchmark) | `vllm bench serve` benchmark tool documentation |
| [SemanticDiff: vLLM CMakeLists.txt SM120](https://app.semanticdiff.com/gh/vllm-project/vllm/commit/789562c28c143201a1d2ca35f7adcdf54ef832e5#CMakeLists.txt) | vLLM SM120 build configuration changes (shared by user) |

### Community Projects
| Project/Link | Relevance |
|--------------|-----------|
| [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | Prebuilt vLLM image for DGX Spark |
| [eugr PR #98](https://github.com/eugr/spark-vllm-docker/pull/98) | Nemotron support — `mamba-cache-mode=align`, Mamba SM121 recognition, CUTLASS 4.4.2 |
| [RobTand/spark-vllm-docker (flashinfer-pr-patching)](https://github.com/RobTand/spark-vllm-docker/tree/flashinfer-pr-patching) | Original E2M1 SM121 patch, FlashInfer K=64 SM120 patch |
| [Avarok-Cybersecurity/dgx-vllm](https://github.com/Avarok-Cybersecurity/dgx-vllm) | DGX Spark NVFP4 optimization documentation (`NVFP4_BREAKTHROUGH_DGX_SPARK.md`) |
| [aliez-ren/vllm-qwen3.5-nvfp4-sm120](https://github.com/aliez-ren/vllm-qwen3.5-nvfp4-sm120) | Qwen3.5 NVFP4 SM120 optimization |
| [JungkwanBan/SPARK_Qwen3.5-122B](https://github.com/JungkwanBan/SPARK_Qwen3.5-122B-A10B-NVFP4) | DGX Spark Qwen3.5 recipe |
| [namake-taro/vllm-custom](https://github.com/namake-taro/vllm-custom) | SM120 custom vLLM build |
| [Reddit: How I got Qwen3.5-397B running at speed](https://www.reddit.com/r/LocalLLaMA/comments/1rtrdsv/55_282_toks_how_i_got_qwen35397b_running_at_speed/) | Qwen3.5 NVFP4 optimization case (Marlin + MTP) |
| [HuggingFace: super_v3_reasoning_parser.py](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) | Nemotron-3-Super dedicated reasoning parser |

### NVIDIA Technical References
| Link | Relevance |
|------|-----------|
| [CUDA PDL Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization) | Why GDC flags are needed — PDL sync |
| [PTX ISA: tcgen05-mma-scale-factor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x) | FP4 MMA scale factor layout (E2M1 related) |
| [CUTLASS Blackwell GeForce GEMM examples](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm) | SM120-specific GEMM example code |
| [TRT-LLM commit 52684d7](https://github.com/NVIDIA/TensorRT-LLM/commit/52684d79f7913973cd9e85f1a3de5c07ef01c039) | MoE SM120 fix commit |
| [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | Mamba2 causal-conv1d native kernel (attempted Triton replacement) |
| [state-spaces/mamba v2.2.4](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/selective_state_update.py) | Original Mamba SSM Triton kernel (source of SM121 async bug) |
| [Stacked DGX Sparks (build.nvidia.com)](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks) | DGX Spark dual-unit connection guide |
| [arxiv: Nemotron-3-Super paper (2512.12087)](https://arxiv.org/abs/2512.12087) | Model architecture paper |
| [arxiv: NemotronH architecture (2501.01005)](https://arxiv.org/abs/2501.01005) | NemotronH hybrid architecture details (Mamba+MoE+Attention) |
| [NVIDIA NeMo: Nemotron-3-Super vLLM cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Super/vllm_cookbook.ipynb) | Official vLLM serving recipe |
| [NVIDIA NeMo: Nemotron-3-Super README](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/super3/README.md) | Model deployment guide |
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) | NVFP4 quantization tool |
| [CUDA Runtime API: cudaFuncSetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html) | MaxDynamicSharedMemorySize configuration reference |
| [PTX ISA: cvt (data conversion)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt) | E2M1 conversion PTX instruction reference |
| [CUTLASS: SM120 BlockScaled Layout docs](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0y_blockscaled_layout.md) | FP4 BlockScaled memory layout |
| [Avarok/dgx-vllm: NVFP4 Breakthrough](https://github.com/Avarok-Cybersecurity/dgx-vllm/blob/main/NVFP4_BREAKTHROUGH.md) | DGX Spark NVFP4 breakthrough documentation |
| [Avarok/dgx-vllm: FlashInfer NVFP4 MoE fix](https://github.com/Avarok-Cybersecurity/dgx-vllm/blob/main/fix_flashinfer_nvfp4_moe_backend.py) | FlashInfer MoE backend fix script |
| [Avarok/dgx-vllm: NVFP4 emulation fix](https://github.com/Avarok-Cybersecurity/dgx-vllm/blob/main/fix_nvfp4_emulation_backend.py) | NVFP4 emulation backend fix |
| [kvcache-ai/custom_flashinfer](https://github.com/kvcache-ai/custom_flashinfer) | SM120 custom FlashInfer build |
| [vLLM docs: torch_compile design](https://docs.vllm.ai/en/v0.10.0/design/v1/torch_compile.html) | torch.compile integration design document |
| [vLLM docs: compilation config](https://docs.vllm.ai/en/v0.10.1.1/api/vllm/config/compilation.html) | compilation_config options reference |
| [vLLM docs: engine args](https://docs.vllm.ai/en/v0.10.2/configuration/engine_args.html) | Complete engine arguments documentation |
| [PyTorch #122706](https://github.com/pytorch/pytorch/issues/122706) | Functionalized tensor `data_ptr()` call issue (TVM DLPack bug related) |

---

## 8. Final Applied Configuration

### docker-compose.yml Key Settings

```yaml
environment:
  FLASHINFER_CUDA_ARCH_LIST: "12.1a"
  FLASHINFER_DISABLE_VERSION_CHECK: "1"
  FLASHINFER_EXTRA_CUDAFLAGS: "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"
  VLLM_KV_CACHE_LAYOUT: "HND"
  VLLM_NVFP4_GEMM_BACKEND: "flashinfer-cutlass"
  VLLM_USE_FLASHINFER_MOE_FP4: "1"
  TORCHINDUCTOR_COMPILE_THREADS: "1"
  MAX_JOBS: "2"

command:
  --model=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
  --tensor-parallel-size=1
  --enable-chunked-prefill
  --max-model-len=131072
  --attention-backend=FLASHINFER
  --enable-prefix-caching
  --mamba-cache-mode=align
  --max-num-batched-tokens=8400
  --load-format=fastsafetensors
  --gpu-memory-utilization=0.8
  --attention-config={"disable_flashinfer_q_quantization":true}
  --compilation-config={"cudagraph_mode":"full_and_piecewise","pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true}}
  --speculative-config={"method":"mtp","num_speculative_tokens":2}
```

---

## 9. Remaining Limitations

### MoE Scatter Bandwidth — Hard Bottleneck

Each of the model's 40 MoE layers must read 22 out of 512 experts per token.

- **Sequential bandwidth:** 241.5 GB/s
- **Scatter read effective bandwidth:** ~57-110 GB/s
- **L2 cache:** 24 MB — cannot cache 512 expert weights
- **Result:** ~60% of decoding time is spent loading MoE weights

### Unsolvable Hardware Constraints

1. **48 SMs** — max_autotune not possible, parallelism limited
2. **101KB SMEM/SM** — large CUTLASS tiles not possible
3. **24MB L2** — MoE expert caching not possible
4. **nsys unsupported** — SM121 hardware tracer limitation

---

## 10. Exhaustive Benchmark Results (144 combinations, `vllm bench serve` official CLI)

> **Run date:** 2026-03-22 ~ 2026-03-23
> **Tool:** `vllm bench serve` (official CLI, [docs](https://docs.vllm.ai/en/latest/benchmarking/cli/#online-benchmark))
> **Dataset:** ShareGPT (real conversation text)
> **Measurement:** rate=1 (single user latency) + rate=inf (max throughput)
> **Combinations:** GEMM(2) × Attention(2) × MTP(3) × CUDAGraph(3) × Fusion(2) × Eager(2) = 144
> **Results:** 84 OK, 60 SKIP (duplicate/impossible combinations auto-skipped), 0 FAIL

### Legend

| Abbreviation | Meaning |
|------|---------|
| **FC** | FlashInfer CUTLASS (native NVFP4 hardware FP4) |
| **Marlin** | Marlin W4A16 (FP4→BF16 dequant then BF16 GEMM) |
| **FI** | FlashInfer Attention (JIT sm_121a) |
| **Triton** | TRITON_ATTN |
| **f&p** | `full_and_piecewise` CUDA graph mode |
| **piece** | `piecewise` CUDA graph mode |
| **none** | No CUDA graph (torch.compile only) |
| **Eager** | `--enforce-eager` (torch.compile + CUDA graph both disabled) |
| **ON/OFF** | `fuse_norm_quant` + `fuse_act_quant` kernel fusion |

### Full Results (Sorted by Single tok/s Descending)

| # | GEMM | Attn | MTP | CUDAGraph | Fuse | Eager | Single tok/s | Max tok/s | TPOT (ms) | P99 ITL (ms) |
|---|------|------|-----|-----------|------|-------|-------------|-----------|-----------|-------------|
| 🥇1 | Marlin | Triton | 2 | f&p | ON | | **46.3** | **63.3** | 116.9 | 486.2 |
| 🥈2 | Marlin | FI | 2 | f&p | ON | | **45.3** | 61.5 | 105.4 | 491.4 |
| 🥉3 | FC | Triton | 2 | f&p | ON | | **44.7** | 44.0 | 106.5 | 468.7 |
| 4 | Marlin | FI | 2 | none | ON | | 43.9 | **62.9** | 134.9 | 800.2 |
| 5 | Marlin | Triton | 2 | — | — | Y | 43.7 | 56.1 | 126.4 | 912.1 |
| 6 | FC | FI | 2 | f&p | OFF | | 42.6 | 47.7 | 109.5 | 471.6 |
| 7 | Marlin | FI | 1 | none | ON | | 42.5 | 62.0 | 133.8 | 556.4 |
| 8 | Marlin | Triton | 2 | none | ON | | 42.0 | 57.9 | 109.3 | 478.3 |
| 9 | Marlin | FI | 1 | f&p | OFF | | 42.0 | 55.8 | 106.4 | 422.5 |
| 10 | Marlin | FI | 1 | f&p | ON | | 42.0 | 58.9 | 116.9 | 452.6 |
| 11 | Marlin | Triton | 1 | none | OFF | | 41.9 | 60.4 | 114.3 | 570.7 |
| 12 | Marlin | Triton | 1 | f&p | ON | | 41.8 | 58.1 | 119.3 | 455.6 |
| 13 | Marlin | FI | 2 | piece | ON | | 41.4 | 51.7 | 123.4 | 491.7 |
| 14 | Marlin | Triton | 1 | piece | ON | | 41.2 | 58.2 | 110.0 | 421.8 |
| 15 | Marlin | FI | 2 | piece | OFF | | 41.1 | 54.5 | 104.7 | 491.1 |
| 16 | Marlin | Triton | 2 | none | OFF | | 40.8 | 60.0 | 138.3 | 1377.6 |
| 17 | Marlin | FI | 1 | none | OFF | | 40.3 | 54.8 | 112.8 | 409.3 |
| 18 | Marlin | FI | 1 | piece | OFF | | 40.1 | 55.6 | 111.2 | 427.0 |
| 19 | Marlin | Triton | 1 | none | ON | | 39.8 | 49.9 | 115.4 | 454.8 |
| 20 | Marlin | FI | 2 | — | — | Y | 39.7 | 55.6 | 154.1 | 969.0 |
| 21 | FC | FI | 1 | none | OFF | | 39.6 | 55.5 | 130.5 | 588.2 |
| 22 | Marlin | FI | 2 | f&p | OFF | | 39.6 | 48.1 | 133.3 | 1081.2 |
| 23 | FC | FI | 2 | none | OFF | | 39.5 | 55.6 | 131.3 | 772.8 |
| 24 | Marlin | FI | 1 | — | — | Y | 39.5 | 55.8 | 124.9 | 672.5 |
| 25 | FC | Triton | 1 | f&p | OFF | | 39.2 | 51.9 | 115.6 | 462.4 |
| 26 | FC | FI | 2 | piece | OFF | | 39.0 | 46.2 | 110.3 | 447.5 |
| 27 | Marlin | Triton | 2 | piece | OFF | | 38.8 | 51.2 | 114.0 | 446.7 |
| 28 | FC | Triton | 1 | none | OFF | | 38.8 | 57.0 | 121.7 | 485.8 |
| 29 | FC | FI | 2 | piece | ON | | 38.6 | 38.1 | 121.2 | 450.9 |
| 30 | FC | Triton | 1 | piece | ON | | 38.4 | 49.5 | 125.4 | 434.9 |
| 31 | FC | FI | 1 | f&p | OFF | | 38.3 | 53.3 | 132.4 | 466.1 |
| 32 | FC | Triton | 1 | piece | OFF | | 37.6 | 52.0 | 132.3 | 439.8 |
| 33 | Marlin | Triton | 2 | f&p | OFF | | 37.0 | 55.9 | 104.6 | 1293.5 |
| 34 | FC | Triton | 1 | — | — | Y | 37.0 | 52.9 | 131.6 | 478.4 |
| 35 | FC | FI | 1 | f&p | ON | | 36.9 | 50.2 | 123.2 | 463.8 |
| 36 | FC | FI | 1 | none | ON | | 36.8 | 54.5 | 133.1 | 539.2 |
| 37 | Marlin | Triton | 1 | piece | OFF | | 36.6 | 56.5 | 122.2 | 428.8 |
| 38 | FC | Triton | 2 | — | — | Y | 36.5 | 58.5 | 130.6 | 816.6 |
| 39 | FC | Triton | 2 | piece | ON | | 36.2 | 46.5 | 122.3 | 451.5 |
| 40 | FC | FI | 1 | piece | OFF | | 36.1 | 52.4 | 131.0 | 436.5 |
| 41 | FC | FI | 2 | — | — | Y | 36.1 | 58.8 | 147.4 | 913.9 |
| 42 | FC | Triton | 1 | none | ON | | 36.0 | 54.4 | 124.5 | 457.6 |
| 43 | FC | Triton | 2 | f&p | OFF | | 35.5 | 46.9 | 123.1 | 460.1 |
| 44 | Marlin | FI | 1 | piece | ON | | 35.4 | 57.5 | 111.1 | 419.5 |
| 45 | Marlin | Triton | 1 | f&p | OFF | | 35.4 | 59.0 | 106.6 | 423.1 |
| 46 | FC | FI | 1 | piece | ON | | 35.1 | 49.8 | 134.4 | 436.0 |
| 47 | FC | Triton | 2 | none | ON | | 34.9 | 60.5 | 134.8 | 768.5 |
| 48 | FC | Triton | 2 | piece | OFF | | 34.5 | 47.8 | 128.0 | 431.9 |
| 49 | FC | FI | 2 | none | ON | | 34.4 | 55.8 | 187.8 | 477.7 |
| 50 | FC | Triton | 2 | none | OFF | | 34.2 | 53.3 | 137.0 | 740.0 |
| 51 | Marlin | FI | 0 | f&p | ON | | 33.4 | 43.4 | 127.3 | 458.5 |
| 52 | FC | FI | 2 | f&p | ON | | 33.2 | 47.7 | 120.3 | 456.4 |
| 53 | Marlin | Triton | 0 | f&p | ON | | 33.2 | 42.7 | 126.3 | 466.3 |
| 54 | FC | Triton | 1 | f&p | ON | | 33.1 | 50.6 | 128.3 | 457.4 |
| 55 | FC | FI | 1 | — | — | Y | 33.0 | 56.0 | 135.6 | 618.2 |
| 56 | Marlin | Triton | 0 | f&p | OFF | | 32.3 | 43.5 | 130.8 | 367.0 |
| 57 | Marlin | FI | 0 | piece | OFF | | 32.2 | 42.9 | 140.2 | 211.6 |
| 58 | Marlin | FI | 0 | piece | ON | | 32.0 | 45.4 | 139.8 | 209.5 |
| 59 | Marlin | Triton | 0 | none | ON | | 32.0 | 46.1 | 138.8 | 219.9 |
| 60 | Marlin | FI | 0 | none | ON | | 31.8 | 41.4 | 137.6 | 217.1 |
| 61 | Marlin | FI | 0 | — | — | Y | 31.7 | 44.8 | 139.2 | 219.5 |
| 62 | Marlin | Triton | 0 | piece | OFF | | 31.6 | 46.3 | 140.2 | 215.0 |
| 63 | Marlin | Triton | 0 | piece | ON | | 31.6 | 45.0 | 139.7 | 217.1 |
| 64 | Marlin | FI | 2 | none | OFF | | 31.0 | 59.4 | 167.5 | 643.9 |
| 65 | Marlin | Triton | 0 | — | — | Y | 30.8 | 43.6 | 141.8 | 226.3 |
| 66 | FC | FI | 0 | f&p | ON | | 30.3 | 42.4 | 149.9 | 305.7 |
| 67 | FC | Triton | 0 | f&p | OFF | | 29.9 | 42.4 | 147.4 | 361.0 |
| 68 | FC | FI | 0 | none | ON | | 29.0 | 43.9 | 143.6 | 365.8 |
| 69 | FC | Triton | 0 | — | — | Y | 28.7 | 41.3 | 143.6 | 299.5 |
| 70 | Marlin | FI | 0 | f&p | OFF | | 28.4 | 43.5 | 135.6 | 493.1 |
| 71 | Marlin | Triton | 2 | piece | ON | | 28.4 | 51.8 | 142.8 | 460.1 |
| 72 | FC | FI | 0 | piece | ON | | 27.6 | 41.9 | 154.8 | 238.2 |
| 73 | FC | FI | 0 | none | OFF | | 27.4 | 43.9 | 126.2 | 216.2 |
| 74 | FC | Triton | 0 | piece | ON | | 27.3 | 42.5 | 144.5 | 362.5 |
| 75 | FC | Triton | 0 | piece | OFF | | 27.3 | 42.3 | 133.2 | 220.5 |
| 76 | Marlin | Triton | 1 | — | — | Y | 26.3 | 54.6 | 132.2 | 637.2 |
| 77 | FC | Triton | 0 | none | ON | | 26.1 | 44.2 | 152.2 | 393.8 |
| 78 | FC | FI | 0 | f&p | OFF | | 26.1 | 42.8 | 149.8 | 514.3 |
| 79 | FC | FI | 0 | — | — | Y | 25.8 | 42.5 | 133.0 | 224.5 |
| 80 | FC | FI | 0 | piece | OFF | | 25.7 | 42.2 | 149.6 | 364.2 |
| 81 | Marlin | Triton | 0 | none | OFF | | 25.4 | 45.4 | 119.9 | 337.5 |
| 82 | FC | Triton | 0 | none | OFF | | 23.7 | 43.4 | 152.5 | 402.6 |
| 83 | FC | Triton | 0 | f&p | ON | | 23.5 | 43.4 | 153.6 | 541.0 |
| 84 | Marlin | FI | 0 | none | OFF | | 21.6 | 44.5 | 136.0 | 407.0 |

### Key Findings

**1. Optimal combination: `Marlin + TRITON_ATTN + MTP=2 + full_and_piecewise + fusion ON` = 46.3 tok/s**

- Runner-up: `Marlin + FlashInfer + MTP=2 + f&p + fusion ON` = 45.3 tok/s (nearly identical)
- FlashInfer CUTLASS best: `FC + TRITON_ATTN + MTP=2 + f&p + fusion ON` = 44.7 tok/s (3rd place)

**2. Marlin > FlashInfer CUTLASS (average +10~15% under same conditions)**

- On SM121's 101KB SMEM, CUTLASS FP4 BlockScaled can only use 1 tile type (128x128x64B) + 2 stages
- All 80 TMA Warp Specialized tactics fail due to SMEM overflow
- Marlin W4A16 uses BF16 tensor cores with fewer SMEM constraints

**3. MTP effect: MTP=0 → MTP=2 gives +35~45% improvement**

| MTP | Average Single tok/s (top 10) | Representative TPOT |
|-----|-------------------------------|---------------------|
| 0 | 31.2 | 137ms |
| 1 | 39.7 | 118ms |
| 2 | 41.6 | 116ms |

**4. CUDA graph mode effect**

| Mode | Average tok/s (MTP=2 combinations) | Notes |
|------|-------------------------------------|-------|
| full_and_piecewise | 40.3 | Optimal (Mamba + MTP compatible) |
| none (torch.compile) | 38.1 | No graph overhead but launch cost |
| piecewise | 37.4 | Slightly slower than f&p |
| enforce-eager | 38.5 | Performs well even without compile |

**5. Fusion effect: ON is +5~10% better than OFF on average**

**6. Attention backend: TRITON_ATTN ≈ FLASHINFER (negligible difference)**

- Both backends appear nearly interchangeably in the top 10
- FlashInfer slightly better on TPOT (105ms vs 117ms), TRITON slightly better on throughput

---

---

## 11. CUTLASS SMEM Optimization (Phase 10, 03-23) ⭐

Analyzed the root cause of why Marlin was 10~15% faster than FlashInfer CUTLASS across the 144 exhaustive benchmarks, and directly modified CUTLASS headers to make **FlashInfer CUTLASS surpass Marlin**.

### Root Cause: CUTLASS SMEM Alignment Waste + StageCountAutoCarveout Calculation Error

`alignas(1024)` was used in `sm120_blockscaled_mma_array_tma.hpp`, causing SMEM alignment waste. Additionally, `StageCountAutoCarveout` did not account for epilogue/pipeline SharedStorage overhead, resulting in undercounting stages (only 2 stages used).

### 4 CUTLASS Patches Applied

| File | Change | Effect |
|------|--------|--------|
| `sm120_blockscaled_mma_array_tma.hpp` | `alignas(1024)` → `alignas(128)` | Eliminate SMEM alignment waste |
| `sm120_blockscaled_mma_builder.inl` | Accurate `ReducedSmemCapacityBytes` calculation | Correct stage count |
| `moe_gemm_tma_ws_launcher_patch.inl` | Force `StageCount<4>` on SM120 | 4 pipeline stages (previously 2) |
| `sm103_blockscaled_gemm_array_tma_warpspecialized.hpp` | `static_assert` sm100→sm120 | SMEM check at compile time |

### Performance Results

| Configuration | Single tok/s | Change |
|---------------|-------------|--------|
| FlashInfer CUTLASS (before patch) | 33.2 | baseline |
| + alignas(128) patch | 39.9 | **+20%** |
| + alignas(128) + StageCount<4> | **48.9** | **+47%** ⭐ |
| Marlin (comparison) | 46.3 | — |

**FlashInfer CUTLASS surpasses Marlin for the first time: 48.9 > 46.3 tok/s (+5.6%)**

### StageCount Sweep Results

| StageCount | Single tok/s | Max tok/s |
|------------|-------------|-----------|
| 2 | 40.05 | 71.00 |
| 3 | 46.15 | 56.68 |
| **4** | **48.9** | — |
| 5 | segfault | — |

### Attempted but Failed CUTLASS Optimizations

| Attempt | Result | Cause |
|---------|--------|-------|
| **128x128x128B tile** | segfault | epilogue 32KB + mainloop 72KB = 104KB > 101KB |
| **256x128x64B tile** | segfault | epilogue **64KB** (M×N×bf16 = 256×128×2) + mainloop 54KB = 118KB > 101KB |
| **StageCount<5>** | segfault | mainloop 92KB + overhead > 101KB |
| **--use_fast_math** | 27.2 tok/s (degraded) | FP4 precision broken, MTP acceptance plummeted |
| **SchedulerPipelineStageCount=2** | 40.76 tok/s (degraded) | Insufficient scheduler pipeline |
| **MTP=3** | 39.84 tok/s (degraded) | 3rd position acceptance 38.67% (too low) |
| **MTP=1** | 47.56 / 74.88 tok/s | Only better on max throughput, worse on single |

> **Reference:** [NVIDIA Forum — SM121 CUTLASS kernel optimization results](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960/2)
> **Reference:** [BTankut/dgx-spark-sglang-moe-configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs) — GB10 MoE Triton config (for FP8 GLM-4.7, not applicable to BF16)

### Additional CUTLASS Optimization Analysis (opt2~7)

| opt | Description | Status | Reason |
|-----|-------------|--------|--------|
| opt2 | Separate SF from TMA barrier (dual-pipeline) | Code ready, not applied | Requires dual-barrier support in PipelineTmaAsync (CUTLASS framework change) |
| opt3 | Reduce Grouped GEMM TensorMap overhead | Analysis complete | TMA descriptor 512B (0.5% of SMEM) — reducing is meaningless |
| opt4 | TMA → cp.async transition | Not possible | SM120 MMA ≠ SM100 UMMA, requires full mainloop rewrite |
| opt5 | K-split accumulation | Not possible | Grouped GEMM TileScheduler does not support K splitting, fused_moe integrates routing+GEMM+activation |
| opt6 | Expert prefetch to L2 | Kernel verified working | Triton prefetch kernel 429 GB/s, MoE integration requires C++ grouped GEMM internal modification |
| opt7 | Warp-specialized redesign (cp.async synchronous) | 1249 lines written | `sm121_sync_mma_array.hpp` completed, CollectiveBuilder integration incomplete |

### BTankut GB10 MoE Triton Config Attempt

Applied the GB10 optimal Triton config from [BTankut/dgx-spark-sglang-moe-configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs) to the Nemotron MTP drafter (BF16 Unquantized MoE).

- **Result:** 39.96 tok/s (-18% degradation vs baseline 48.9)
- **Cause:** BTankut config is for **FP8 MoE (E=160, N=384, GLM-4.7)**. BF16 has 2x data size so BLOCK_SIZE_K=256 exceeds SMEM
- **Conclusion:** Individual tuning required per model/dtype

### GPT-OSS-120B Comparison Benchmark

| Model | Architecture | Random single | Random max | Real text | Peak |
|-------|-------------|---------------|------------|-----------|------|
| **Nemotron-3-Super** | Mamba+MoE+Attn (88 layers) | 48.0 tok/s | 53.0 tok/s | 21.9 tok/s | 63.6 tok/s |
| **GPT-OSS-120B** | Transformer (36 layers) | 101.9 tok/s | 159.4 tok/s | 34.8 tok/s | 200 tok/s |

GPT-OSS is 2~3x faster because: half the layers, no MoE/Mamba, no scatter reads

---

## 12. Final Benchmark Results (All Optimizations Applied, 03-23)

> Server configuration: FlashInfer CUTLASS FP4 + StageCount<4> + alignas(128) + MTP=2 + CUDA graph full_and_piecewise + fuse_norm_quant + fuse_act_quant
> vLLM version: 0.17.2rc1.dev162
> Benchmark: `vllm bench serve` (official CLI)

### ShareGPT (Real Text Conversation)

| Metric | rate=1 (20 prompts) | rate=inf (50 prompts) |
|--------|--------------------|-----------------------|
| **Output tok/s** | **49.93** | **63.57** |
| Total tok/s | 102.88 | 138.29 |
| Mean TPOT | 152.65 ms | 155.58 ms |
| P99 ITL | 628.92 ms | 599.46 ms |
| MTP Acceptance | 52.07% | 55.51% |
| Mean Acceptance Length | 2.04 | 2.11 |

### Random Dataset (input=128, output=256)

| Metric | rate=1 (20 prompts) | rate=inf (50 prompts) |
|--------|--------------------|-----------------------|
| **Output tok/s** | **47.98** | **52.98** |
| Total tok/s | 71.97 | 79.46 |
| Mean TPOT | 179.18 ms | 188.67 ms |
| P99 ITL | 399.32 ms | 400.52 ms |
| MTP Acceptance | 15.99% | 15.93% |

> MTP acceptance ~16% on random dataset is low (no predictable patterns)

### Real Text (curl, max_tokens=512)

| Metric | Single Request | 5 Concurrent |
|--------|---------------|--------------|
| **Average tok/s** | **21.9** | — |
| Min / Max | 20.3 / 22.9 | 10.3 / 12.2 |
| **Aggregate tok/s** | — | **51.3** |

---

## 13. Summary

| Metric | Value |
|--------|-------|
| **Optimal Combination** | **FlashInfer CUTLASS + StageCount<4> + MTP=2 + f&p + fusion** |
| **ShareGPT Single Request** | **49.93 tok/s** |
| **ShareGPT Max Throughput** | **63.57 tok/s** |
| **Real Text Single Request** | **21.9 tok/s** |
| **Real Text 5 Concurrent** | **51.3 tok/s** |
| **TPOT** | **152.65 ms** |
| **Improvement Over Previous** | Marlin 46.3 → FC+SMEM 49.93 (+7.8%) |
| **Improvement Over Baseline** | 12.0 → 49.93 (+316%) |
| **Improvement Over Initial** | 2.5 → 49.93 (+1897%) |
| **Exhaustive Benchmarks** | 144 combinations (84 OK, 60 SKIP, 0 FAIL) |
| **CUTLASS Patches** | 4 headers + 1 experimental kernel (sm121_sync_mma_array.hpp) |
| **Runtime Patches** | 5 files |
| **External Issues/PRs Referenced** | 30+ items |
| **Hard Bottleneck** | MoE scatter bandwidth (~110 GB/s effective) |
