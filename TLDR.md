# TL;DR: Just Use Marlin on DGX Spark

I spent 4 days optimizing Nemotron-3-Super-120B-A12B-NVFP4 on DGX Spark (GB10, SM121) with vLLM, testing **144 different configuration combinations** and patching CUTLASS kernel headers to squeeze out every last tok/s from FlashInfer's native NVFP4 path.

**The conclusion: just use Marlin. The performance difference is negligible.**

## The Numbers

| Configuration | Single (tok/s) | Concurrent (tok/s) | Patches Required |
|--------------|---------------|-------------------|-----------------|
| **Marlin (zero CUTLASS patches)** | **46.3** | **63.3** | 0 CUTLASS patches |
| FlashInfer CUTLASS (4 CUTLASS patches + SMEM opt) | 49.9 | 63.6 | 4 files, ~200 lines |

That's a **7.8% difference** on single requests, and **essentially identical on concurrent throughput**. In real-world chat usage (512-token responses), both land at roughly **~22 tok/s** because the bottleneck shifts to MoE scatter bandwidth and MTP acceptance rate — neither of which cares about the GEMM backend.

*Note: FC numbers are from the final benchmark after all CUTLASS SMEM patches (alignas(128) + StageCount<4>). Marlin numbers are from the 144-combo exhaustive benchmark — Marlin doesn't use CUTLASS, so the SMEM patches don't affect it.*

## Why Not Bother with FlashInfer CUTLASS NVFP4

1. **SM121 has only 101KB SMEM** (vs 228KB on SM100). This doesn't just slow things down — it **fundamentally breaks** the native FP4 GEMM path:
   - Out of the box, FlashInfer CUTLASS FP4 MoE **doesn't work at all** on SM121. The autotuner tries 80+ TMA Warp Specialized tactics and **every single one fails** with `Error Internal` (SMEM overflow at `cudaFuncSetAttribute`)
   - Only one tile shape fits: 128×128×64B (the smallest). All larger tiles are physically impossible — the **epilogue alone** needs 64KB for M=256 (more than half the 101KB budget)
   - Maximum 4 pipeline stages (SM100 gets 10+), meaning less latency hiding and lower throughput
   - Without 4 custom CUTLASS header patches, the kernel computes **wrong stage counts** and crashes at runtime with segfaults
   - Even the dense (non-MoE) FP4 GEMM initially produced **garbage output** (all zeros / NaN) on SM121 due to E2M1 instruction incompatibility — required additional software fallback patches

2. **You need 4 CUTLASS header patches** to make it work at all:
   - `alignas(1024)` → `alignas(128)` (SMEM alignment waste)
   - `StageCount<4>` forced (auto-carveout miscalculates)
   - `static_assert` fix (sm100 → sm120 SMEM capacity)
   - `ReducedSmemCapacityBytes` correction (TensorMap overhead)

   Without these patches, FlashInfer CUTLASS either **crashes with a segfault** or runs at **33 tok/s** — 30% slower than Marlin. And even getting to the crash-but-sometimes-works state required 5 additional runtime patches for SM121 compatibility.

3. **The real bottleneck is MoE scatter bandwidth, not GEMM**. This model has 512 experts with 22 active per token across 40 MoE layers. Each decode step reads ~22 scattered expert weight blocks from DRAM. GB10's 24MB L2 cache can't buffer this. Effective bandwidth drops from 241 GB/s (sequential) to ~57-110 GB/s (scattered). No amount of GEMM kernel optimization fixes this.

4. **MTP acceptance rate dominates real-world speed**. With MTP=2 speculative decoding, acceptance rate on real text is 66-92%. On synthetic benchmarks (random tokens), it's only 16-19%. The GEMM backend doesn't affect acceptance rate — it's determined by the model's MTP head predictions.

5. **Marlin just works out of the box**. No patches, no JIT rebuild (saves 5+ minutes on cold start), no risk of SMEM overflow segfaults. Set `VLLM_TEST_FORCE_FP8_MARLIN=1` and go.

## What Actually Matters

If you want to speed up Nemotron-3-Super on DGX Spark, focus on these instead:

| Optimization | Impact | Complexity |
|-------------|--------|-----------|
| **MTP speculative decoding (tokens=2)** | **+33%** | 1 config line |
| **CUDA graph full_and_piecewise** | **+11%** | 3 runtime patches |
| **fuse_norm_quant + fuse_act_quant** | **+3%** | 1 config line |
| **prefix caching + mamba_cache_mode=align** | Memory savings | 2 config lines |

These optimizations are GEMM-backend-agnostic and give you the same benefit whether you use Marlin or FlashInfer CUTLASS.

## Full Details

See the [optimization report](OPTIMIZATION_REPORT.md) for the complete 1400-line analysis including:
- 144-combination exhaustive benchmark (6 axes: GEMM × Attention × MTP × CUDAGraph × Fusion × Eager)
- Every failed attempt documented with error messages and root causes
- CUTLASS SMEM budget analysis down to individual bytes
- 30+ external references (GitHub issues, PRs, NVIDIA forum posts)

**Repo:** [rmagur1203/vllm-dgx-spark](https://github.com/rmagur1203/vllm-dgx-spark)
