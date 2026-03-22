# Nemotron-3-Super-120B on DGX Spark (SM121)

NVIDIA DGX Spark (GB10, SM121, 128GB unified memory)에서
**Nemotron-3-Super-120B-A12B-NVFP4**를 vLLM으로 서빙하기 위한 재현 가능한 설정입니다.

**성능:** 단일 요청 **22 tok/s**, 5개 동시 요청 **50 tok/s** aggregate

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
├── Dockerfile                  # Multi-stage: FlashInfer + vLLM 소스 빌드
├── docker-compose.yml          # 서비스 정의 (drop-caches + nemotron)
├── .env                        # HF_TOKEN, API key (gitignore됨)
├── .env.example                # 환경변수 템플릿
├── patches/
│   ├── build/                  # Docker 이미지 빌드 시 적용되는 패치
│   │   ├── flashinfer_k64_sm120_v442.patch   # K=64 MoE 타일 (SMEM 최적화)
│   │   ├── flashinfer_e2m1_sm121.patch       # CUTLASS E2M1 SM121 제외
│   │   ├── flashinfer_cache.patch            # cubin 캐시 재활용
│   │   ├── e2m1_nvfp4_sm121.patch            # vLLM E2M1 소프트웨어 변환
│   │   └── fix_quantization_utils_sm121.py   # TRT-LLM quantization E2M1 fix
│   └── runtime/                # 컨테이너 실행 시 volume mount되는 패치
│       ├── tvm_build_patch.py                # TVM DLPack has_storage() 수정
│       ├── cutlass_heuristic_patch.cpp       # MoE CUTLASS 타일 크기 제한
│       ├── vllm_flashinfer_utils_patch.py    # SM121 TRT-LLM attention 지원
│       ├── flashinfer_backend_patch.py       # HND layout + XQA causal mask
│       └── mamba_mixer2_patch.py             # Mamba Triton sync workaround
├── scripts/
│   ├── prebuild_flashinfer_jit.py  # JIT 사전 빌드 (모델 로딩 전 실행)
│   └── super_v3_reasoning_parser.py # Nemotron reasoning 파서
└── cache/                      # 영구 캐시 (HF 모델, JIT, torch.compile)
```

## 빠른 시작

### 1. 환경 설정

```bash
cd vllm-dgx-spark
cp .env.example .env
# .env 파일에 HF_TOKEN과 VLLM_API_KEY 입력
```

### 2. 이미지 빌드

```bash
docker compose build nemotron-3-super
```

> ⏱ 첫 빌드: ~30-60분 (FlashInfer + vLLM 소스 컴파일)
> 이후 재빌드: ~5분 (Docker layer cache + ccache 활용)

### 3. 서버 시작

```bash
docker compose up -d
```

> ⏱ 첫 실행: ~25분 (FlashInfer JIT prebuild + 모델 다운로드)
> 이후 재시작: ~7분 (JIT 캐시 + 모델 캐시 재사용)

### 4. 테스트

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $(grep VLLM_API_KEY .env | cut -d= -f2)" \
  -d '{
    "model": "nemotron-3-super",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## 아키텍처

### 왜 이렇게 복잡한가?

DGX Spark의 GB10 GPU는 **SM 12.1** — Blackwell 아키텍처이지만 데스크톱 SM120(RTX 5090)과 다른 점이 많습니다:

| 제약 | SM120 (RTX 5090) | SM121 (DGX Spark) | 영향 |
|------|------------------|-------------------|------|
| SMEM | 228 KB | **101 KB** | CUTLASS 큰 타일 사용 불가 |
| SM 수 | 170 | **48** | max_autotune 불가 |
| E2M1 PTX | ✅ | **❌** | 소프트웨어 변환 필요 |
| TRT-LLM attention | ✅ | **❌** | 패치로 우회 |
| Triton Mamba | ✅ | **❌ (async bug)** | sync 워크어라운드 |
| 메모리 | 독립 GDDR7 | **통합 LPDDR5X** | cudaMemGetInfo 부정확 |

### 패치 설명

#### 빌드 시 패치 (`patches/build/`)

| 패치 | 문제 | 해결 |
|------|------|------|
| `flashinfer_k64_sm120_v442.patch` | K=128 타일이 101KB SMEM 초과 | K=64 타일 추가 (7-11 pipeline stages) |
| `flashinfer_e2m1_sm121.patch` | SM121에 FP4 E2M1 PTX 명령어 없음 | CUTLASS에서 SM121 제외 |
| `e2m1_nvfp4_sm121.patch` | vLLM의 FP4 커널에도 동일 PTX 문제 | 소프트웨어 E2M1 변환 추가 |
| `fix_quantization_utils_sm121.py` | TRT-LLM quantization의 E2M1 | 3개 함수에 SW fallback |

#### 런타임 패치 (`patches/runtime/`)

| 패치 | 문제 | 해결 |
|------|------|------|
| `tvm_build_patch.py` | FlashInfer TVM FFI가 storage 없는 tensor에 `data_ptr()` 호출 | `has_storage()` 체크 + `clone()` |
| `cutlass_heuristic_patch.cpp` | 128x128x128B 이상 타일이 SMEM 초과 | `CtaShape128x128x64B`만 허용 |
| `vllm_flashinfer_utils_patch.py` | `supports_trtllm_attention()=False` → CUDA graph FULL 불가 | SM120 family 지원 추가 |
| `flashinfer_backend_patch.py` | HND 레이아웃 미지원 + spec decode causal mask 없음 | SM120 HND 반환 + mask 생성 |
| `mamba_mixer2_patch.py` | SM121 Triton PTX codegen 메모리 배리어 누락 | `torch.cuda.synchronize()` 삽입 |

## 성능 튜닝 가이드

### 현재 최적 설정

| 설정 | 값 | 이유 |
|------|-----|------|
| `num_speculative_tokens` | 2 | 3은 acceptance 너무 낮음, 1은 이점 적음 |
| `cudagraph_mode` | `full_and_piecewise` | Mamba 레이어에 FULL graph 필수 |
| `gpu-memory-utilization` | 0.8 | 0.9면 CUDA graph 캡처 시 OOM |
| `mamba-cache-mode` | `align` | `all`이면 MTP spec decode와 충돌 |
| `fuse_norm_quant` | true | LayerNorm→양자화 퓨전 (+3%) |
| `fuse_act_quant` | true | 활성화→양자화 퓨전 (+3%) |

### 시도해봤지만 효과 없는 것들

| 설정 | 결과 |
|------|------|
| `fuse_attn_quant: true` | FP8 query 필요 → SM121 XQA와 호환 불가 |
| `max_autotune: true` | SM 48개 부족 |
| `num_speculative_tokens: 3+` | 3번째 위치 acceptance 48-59% |
| THP (Transparent Huge Pages) | 통합 메모리에서 효과 없음 |

## 벤치마크 결과

| 시나리오 | 처리량 |
|----------|--------|
| 단일 요청 (실제 텍스트, 512 tok) | **22.1 tok/s** (avg), 24.8 (peak) |
| 5개 동시 요청 | **49.7 tok/s** aggregate |
| vllm bench serve (random, rate=inf) | **42.6 tok/s** output |

### MTP Acceptance Rate (실제 텍스트)

| 위치 | Acceptance |
|------|-----------|
| Position 0 | 79-97% |
| Position 1 | 53-88% |
| Mean acceptance length | 2.3-2.8 |

## 문제 해결

### 서버가 OOM으로 죽는 경우

```bash
# gpu-memory-utilization 낮추기
# docker-compose.yml에서 --gpu-memory-utilization=0.75 로 변경

# 또는 swap 확장
sudo fallocate -l 32G /swap.img
sudo mkswap /swap.img && sudo swapon /swap.img
```

### JIT 컴파일이 너무 오래 걸리는 경우

첫 실행 시 FlashInfer MoE 커널 컴파일에 ~25분 소요. `cache/` 볼륨이 유지되면
재시작 시 캐시에서 로딩 (~10초).

```bash
# 캐시 초기화가 필요한 경우
rm -rf cache/flashinfer/.moe_patched
```

### CUDA error: illegal instruction

Mamba Triton sync 워크어라운드가 적용되지 않은 경우:
- `mamba_mixer2_patch.py`가 올바르게 volume mount되었는지 확인
- 컨테이너 내에서 `python3 -c "from vllm.model_executor.layers.mamba.mamba_mixer2 import *"` 실행하여 패치 확인

## 참고 자료

- [OPTIMIZATION_REPORT.md](../vllm/OPTIMIZATION_REPORT.md) — 전체 최적화 과정 상세 기록 (1,100+ 줄)
- [CUTLASS #3096](https://github.com/NVIDIA/cutlass/issues/3096) — SM120 FP4 GEMM 커뮤니티 문서
- [FlashInfer PR #2786](https://github.com/flashinfer-ai/flashinfer/pull/2786) — K=64 타일 추가
- [TensorRT-LLM PR #5823](https://github.com/NVIDIA/TensorRT-LLM/pull/5823) — MoE SM120 수정
- [vLLM #37431](https://github.com/vllm-project/vllm/issues/37431) — SM12x Mamba Triton bug
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — 커뮤니티 DGX Spark 프로젝트
