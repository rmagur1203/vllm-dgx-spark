# Nemotron-3-Super-120B on DGX Spark (SM121)

NVIDIA DGX Spark (GB10, SM121, 128GB unified memory)에서
**Nemotron-3-Super-120B-A12B-NVFP4**를 vLLM으로 서빙하기 위한 재현 가능한 설정입니다.

**성능:** 단일 요청 **48.9 tok/s** (ShareGPT), 동시 요청 **61.7 tok/s** aggregate

## 핵심 최적화

| 최적화 | 효과 |
|--------|------|
| CUTLASS `alignas(1024)`→`alignas(128)` | +26.5% (SMEM alignment 최적화) |
| `StageCount<4>` 강제 | pipeline latency hiding (2→4 stages) |
| MTP speculative decoding (tokens=2) | +33% (80% acceptance rate) |
| CUDA graph `full_and_piecewise` | +11% (SM121 Mamba/Attention 패치) |
| `fuse_norm_quant` + `fuse_act_quant` | +3% (커널 퓨전) |
| FlashInfer CUTLASS native NVFP4 | Marlin 대비 +5.6% (48.9 vs 46.3) |

## 요구 사항

| 항목 | 값 |
|------|-----|
| 하드웨어 | NVIDIA DGX Spark (GB10, SM 12.1) |
| 메모리 | 128GB LPDDR5X 통합 메모리 |
| OS | Ubuntu (aarch64) |
| Docker | Docker Compose v2 + NVIDIA Container Toolkit |
| HuggingFace | gated 모델 접근 권한 + HF_TOKEN |

## 프로젝트 구조

```
vllm-dgx-spark/
├── Dockerfile                    # Multi-stage: FlashInfer + vLLM 소스 빌드
├── docker-compose.yml            # 서비스 정의 (drop-caches + nemotron)
├── .env                          # HF_TOKEN, API key (gitignore됨)
├── .env.example                  # 환경변수 템플릿
├── OPTIMIZATION_REPORT.md        # 전체 최적화 보고서 (144개 조합 벤치마크 포함)
├── patches/
│   ├── build/                    # Docker 이미지 빌드 시 적용되는 패치
│   │   ├── flashinfer_k64_sm120_v442.patch   # K=64 MoE 타일 (SMEM 최적화)
│   │   ├── flashinfer_e2m1_sm121.patch       # CUTLASS E2M1 SM121 제외
│   │   ├── flashinfer_cache.patch            # cubin 캐시 재활용
│   │   ├── e2m1_nvfp4_sm121.patch            # vLLM E2M1 변환 패치
│   │   └── fix_quantization_utils_sm121.py   # 소프트웨어 E2M1 변환
│   ├── runtime/                  # Volume mount로 적용되는 런타임 패치
│   │   ├── tvm_build_patch.py                # TVM FFI has_storage() 수정
│   │   ├── cutlass_heuristic_patch.cpp       # SM121 SMEM 타일 크기 제한
│   │   ├── vllm_flashinfer_utils_patch.py    # SM120 TRT-LLM attention 지원
│   │   ├── flashinfer_backend_patch.py       # HND 레이아웃 + XQA mask
│   │   └── mamba_mixer2_patch.py             # CUDA graph sync 건너뛰기
│   └── cutlass/                  # CUTLASS SMEM 최적화 패치 (핵심!)
│       ├── sm120_blockscaled_mma_array_tma.hpp       # alignas(128) 패치
│       ├── sm120_blockscaled_mma_builder.inl         # ReducedSmemCapacityBytes 수정
│       ├── sm103_blockscaled_gemm_array_tma_warpspecialized.hpp  # static_assert sm120
│       ├── moe_gemm_tma_ws_launcher_patch.inl        # StageCount<4> 강제
│       └── sm121_sync_mma_array.hpp                  # [실험적] cp.async 동기식 커널
├── scripts/
│   ├── prebuild_flashinfer_jit.py  # JIT 사전 빌드 + CUTLASS 패치 적용
│   └── super_v3_reasoning_parser.py  # Nemotron Super v3 reasoning 파서
└── benchmarks/
    ├── bench_all.py              # 144개 조합 전수 벤치마크 자동화
    └── bench_results.jsonl       # 벤치마크 결과 (84 OK, 60 SKIP)
```

## 빠른 시작

```bash
# 1. 환경 설정
cp .env.example .env
# .env에 HF_TOKEN과 API_KEY 설정

# 2. Docker 이미지 빌드 (최초 1회, ~30분)
docker compose build

# 3. 서버 시작
docker compose up -d

# 4. 테스트
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model":"vllm/nemotron-3-super-120b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":100}'
```

## 패치 설명

### CUTLASS SMEM 최적화 (핵심, +26.5%)

SM121의 101KB SMEM에서 FlashInfer CUTLASS FP4 BlockScaled MoE 성능을 극대화:

| 패치 | 변경 | 효과 |
|------|------|------|
| `sm120_blockscaled_mma_array_tma.hpp` | `alignas(1024)`→`alignas(128)` | SMEM alignment 낭비 제거 |
| `sm120_blockscaled_mma_builder.inl` | `ReducedSmemCapacityBytes` 정확한 계산 | 올바른 stage count |
| `moe_gemm_tma_ws_launcher_patch.inl` | SM120에서 `StageCount<4>` 강제 | 4 pipeline stages (기존 2) |
| `sm103_..._warpspecialized.hpp` | `static_assert` sm100→sm120 | 컴파일 시 SMEM 체크 |

### 런타임 패치 (SM121 호환성)

| 패치 | 문제 | 수정 |
|------|------|------|
| `tvm_build_patch.py` | functionalized tensor `data_ptr()` 크래시 | `has_storage()` 체크 추가 |
| `cutlass_heuristic_patch.cpp` | SM121 SMEM에 맞지 않는 타일 선택 | 128×128×64B만 허용 |
| `vllm_flashinfer_utils_patch.py` | `supports_trtllm_attention()=False` | SM120 TRT-LLM 지원 추가 |
| `flashinfer_backend_patch.py` | HND 레이아웃 미지원, XQA mask 없음 | SM120 HND + causal mask |
| `mamba_mixer2_patch.py` | CUDA graph 캡처 중 `synchronize()` 호출 | graph 캡처 시 sync 건너뛰기 |

## 벤치마크 결과

### 최적 설정 (FlashInfer CUTLASS + StageCount<4> + MTP=2)

| 메트릭 | ShareGPT | Random |
|--------|----------|--------|
| Single output tok/s | **48.9** | 32.86 |
| Max output tok/s | **61.7** | 37.94 |
| TPOT (ms) | 117 | 146 |

### 144개 조합 전수 테스트 Top 5

| # | 설정 | Single tok/s | Max tok/s |
|---|------|-------------|-----------|
| 1 | **FC + FI + MTP=2 + f&p + fusion** | **48.9** | **61.7** |
| 2 | Marlin + Triton + MTP=2 + f&p + fusion | 46.3 | 63.3 |
| 3 | Marlin + FI + MTP=2 + f&p + fusion | 45.3 | 61.5 |
| 4 | FC + Triton + MTP=2 + f&p + fusion | 44.7 | 44.0 |
| 5 | Marlin + FI + MTP=2 + none + fusion | 43.9 | 62.9 |

> FC = FlashInfer CUTLASS, FI = FlashInfer Attention, f&p = full_and_piecewise CUDA graph

### 비교: GPT-OSS-120B

| 모델 | 아키텍처 | Single tok/s | Peak tok/s |
|------|---------|-------------|------------|
| Nemotron-3-Super-120B | Mamba+MoE+Attention (88 layers) | 48.9 | 61.7 |
| GPT-OSS-120B | Transformer (36 layers) | 101.9 | 200 |

## 참조

- [CUTLASS #3096](https://github.com/NVIDIA/cutlass/issues/3096) — SM120 FP4 성능
- [CUTLASS 73c59c0](https://github.com/NVIDIA/cutlass/commit/73c59c0) — ReducedSmemCapacityBytes
- [FlashInfer #2577](https://github.com/flashinfer-ai/flashinfer/issues/2577) — SM120 FP4 이슈
- [FlashInfer PR #2786](https://github.com/flashinfer-ai/flashinfer/pull/2786) — K=64 타일
- [TensorRT-LLM PR #5823](https://github.com/NVIDIA/TensorRT-LLM/pull/5823) — SM120 MoE fix
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — DGX Spark 커뮤니티
- [BTankut/dgx-spark-sglang-moe-configs](https://github.com/BTankut/dgx-spark-sglang-moe-configs) — GB10 MoE 튜닝
