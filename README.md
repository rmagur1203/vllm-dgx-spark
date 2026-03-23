# Nemotron-3-Super-120B on DGX Spark (SM121)

A reproducible setup for serving **Nemotron-3-Super-120B-A12B-NVFP4** with vLLM
on NVIDIA DGX Spark (GB10, SM121, 128GB unified memory).

**Performance:** Single request **48.9 tok/s** (ShareGPT), concurrent **61.7 tok/s** aggregate

## Key Optimizations

| Optimization | Effect |
|-------------|--------|
| CUTLASS `alignas(1024)`→`alignas(128)` | +26.5% (SMEM alignment optimization) |
| Forced `StageCount<4>` | Pipeline latency hiding (2→4 stages) |
| MTP speculative decoding (tokens=2) | +33% (80% acceptance rate) |
| CUDA graph `full_and_piecewise` | +11% (SM121 Mamba/Attention patches) |
| `fuse_norm_quant` + `fuse_act_quant` | +3% (kernel fusion) |
| FlashInfer CUTLASS native NVFP4 | +5.6% vs Marlin (48.9 vs 46.3) |

## Requirements

| Item | Value |
|------|-------|
| Hardware | NVIDIA DGX Spark (GB10, SM 12.1) |
| Memory | 128GB LPDDR5X unified memory |
| OS | Ubuntu (aarch64) |
| Docker | Docker Compose v2 + NVIDIA Container Toolkit |
| HuggingFace | Gated model access + HF_TOKEN |

## Project Structure

```
vllm-dgx-spark/
├── Dockerfile                    # Multi-stage: FlashInfer + vLLM source build
├── docker-compose.yml            # Service definitions (drop-caches + nemotron)
├── .env                          # HF_TOKEN, API key (gitignored)
├── .env.example                  # Environment variable template
├── OPTIMIZATION_REPORT.md        # Full optimization report (144-combo benchmark included)
├── patches/
│   ├── build/                    # Patches applied during Docker image build
│   │   ├── flashinfer_k64_sm120_v442.patch   # K=64 MoE tile (SMEM optimization)
│   │   ├── flashinfer_e2m1_sm121.patch       # CUTLASS E2M1 SM121 exclusion
│   │   ├── flashinfer_cache.patch            # cubin cache reuse
│   │   ├── e2m1_nvfp4_sm121.patch            # vLLM E2M1 conversion patch
│   │   └── fix_quantization_utils_sm121.py   # Software E2M1 conversion
│   ├── runtime/                  # Runtime patches applied via volume mount
│   │   ├── tvm_build_patch.py                # TVM FFI has_storage() fix
│   │   ├── cutlass_heuristic_patch.cpp       # SM121 SMEM tile size restriction
│   │   ├── vllm_flashinfer_utils_patch.py    # SM120 TRT-LLM attention support
│   │   ├── flashinfer_backend_patch.py       # HND layout + XQA mask
│   │   └── mamba_mixer2_patch.py             # CUDA graph sync skip
│   └── cutlass/                  # CUTLASS SMEM optimization patches (core!)
│       ├── sm120_blockscaled_mma_array_tma.hpp       # alignas(128) patch
│       ├── sm120_blockscaled_mma_builder.inl         # ReducedSmemCapacityBytes fix
│       ├── sm103_blockscaled_gemm_array_tma_warpspecialized.hpp  # static_assert sm120
│       ├── moe_gemm_tma_ws_launcher_patch.inl        # Forced StageCount<4>
│       └── sm121_sync_mma_array.hpp                  # [Experimental] cp.async synchronous kernel
├── scripts/
│   ├── prebuild_flashinfer_jit.py  # JIT pre-build + CUTLASS patch application
│   └── super_v3_reasoning_parser.py  # Nemotron Super v3 reasoning parser
└── benchmarks/
    ├── bench_all.py              # 144-combo exhaustive benchmark automation
    └── bench_results.jsonl       # Benchmark results (84 OK, 60 SKIP)
```

## Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env to set HF_TOKEN and API_KEY

# 2. Build Docker image (first time only, ~30 min)
docker compose build

# 3. Start server
docker compose up -d

# 4. Test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model":"vllm/nemotron-3-super-120b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

## Patch Descriptions

### CUTLASS SMEM Optimization (Core, +26.5%)

Maximizes FlashInfer CUTLASS FP4 BlockScaled MoE performance within SM121's 101KB SMEM:

| Patch | Change | Effect |
|-------|--------|--------|
| `sm120_blockscaled_mma_array_tma.hpp` | `alignas(1024)`→`alignas(128)` | Eliminates SMEM alignment waste |
| `sm120_blockscaled_mma_builder.inl` | Accurate `ReducedSmemCapacityBytes` calculation | Correct stage count |
| `moe_gemm_tma_ws_launcher_patch.inl` | Force `StageCount<4>` for SM120 | 4 pipeline stages (was 2) |
| `sm103_..._warpspecialized.hpp` | `static_assert` sm100→sm120 | Compile-time SMEM check |

### Runtime Patches (SM121 Compatibility)

| Patch | Problem | Fix |
|-------|---------|-----|
| `tvm_build_patch.py` | Functionalized tensor `data_ptr()` crash | Added `has_storage()` check |
| `cutlass_heuristic_patch.cpp` | Tiles exceeding SM121 SMEM selected | Only allow 128×128×64B |
| `vllm_flashinfer_utils_patch.py` | `supports_trtllm_attention()=False` | Added SM120 TRT-LLM support |
| `flashinfer_backend_patch.py` | HND layout unsupported, missing XQA mask | SM120 HND + causal mask |
| `mamba_mixer2_patch.py` | `synchronize()` called during CUDA graph capture | Skip sync during capture |

## Benchmark Results

### Optimal Configuration (FlashInfer CUTLASS + StageCount<4> + MTP=2)

| Metric | ShareGPT | Random |
|--------|----------|--------|
| Single output tok/s | **48.9** | 32.86 |
| Max output tok/s | **61.7** | 37.94 |
| TPOT (ms) | 117 | 146 |

### 144-Combo Exhaustive Test Top 5

| # | Configuration | Single tok/s | Max tok/s |
|---|--------------|-------------|-----------|
| 1 | **FC + FI + MTP=2 + f&p + fusion** | **48.9** | **61.7** |
| 2 | Marlin + Triton + MTP=2 + f&p + fusion | 46.3 | 63.3 |
| 3 | Marlin + FI + MTP=2 + f&p + fusion | 45.3 | 61.5 |
| 4 | FC + Triton + MTP=2 + f&p + fusion | 44.7 | 44.0 |
| 5 | Marlin + FI + MTP=2 + none + fusion | 43.9 | 62.9 |

> FC = FlashInfer CUTLASS, FI = FlashInfer Attention, f&p = full_and_piecewise CUDA graph

### Comparison: GPT-OSS-120B

| Model | Architecture | Single tok/s | Peak tok/s |
|-------|-------------|-------------|------------|
| Nemotron-3-Super-120B | Mamba+MoE+Attention (88 layers) | 48.9 | 61.7 |
| GPT-OSS-120B | Transformer (36 layers) | 101.9 | 200 |

## References

- [CUTLASS #3096](https://github.com/NVIDIA/cutlass/issues/3096) — SM120 FP4 performance
- [CUTLASS 73c59c0](https://github.com/NVIDIA/cutlass/commit/73c59c0) — ReducedSmemCapacityBytes
- [FlashInfer #2577](https://github.com/flashinfer-ai/flashinfer/issues/2577) — SM120 FP4 issues
- [FlashInfer PR #2786](https://github.com/flashinfer-ai/flashinfer/pull/2786) — K=64 tiles
- [TensorRT-LLM PR #5823](https://github.com/NVIDIA/TensorRT-LLM/pull/5823) — SM120 MoE fix
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — DGX Spark community
- [BTankut/dgx-spark-sglang-moe-configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs) — GB10 MoE tuning
