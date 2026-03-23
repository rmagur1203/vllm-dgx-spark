# Nemotron-3-Super-120B on DGX Spark (SM121) — 최적화 종합 보고서

> **기간:** 2026-03-18 ~ 2026-03-21 (4일)
> **목표:** NVIDIA DGX Spark에서 Nemotron-3-Super-120B-A12B-NVFP4의 추론 성능 최대화
> **최종 결과:** 단일 요청 평균 **22.1 tok/s**, 5개 동시 요청 aggregate **49.7 tok/s**

---

## 1. 하드웨어 및 모델 개요

### 하드웨어: NVIDIA DGX Spark (GB10)

| 항목 | 값 |
|------|-----|
| GPU | NVIDIA GB10 (Blackwell) |
| Compute Capability | SM 12.1 (sm_121a) |
| SM 수 | 48개 (vs B200의 132개) |
| 메모리 | 128GB LPDDR5X 통합 메모리 (CPU+GPU 공유) |
| 메모리 대역폭 | 241.5 GB/s (순차) |
| L2 캐시 | 24 MB |
| SMEM / SM | 101 KB (vs SM100의 228KB) |
| GPU 클럭 | 3003 MHz |

### 모델: Nemotron-3-Super-120B-A12B-NVFP4 (NemotronH 아키텍처)

| 항목 | 값 |
|------|-----|
| 총 파라미터 | 120B |
| 활성 파라미터 | 12B / token |
| 레이어 | 88개 (40 Mamba + 40 MoE + 8 Attention) |
| 레이어 패턴 | `MEMEMEM*EMEMEMEM*...` |
| MTP 레이어 패턴 | `*E` (1 Attention + 1 MoE) |
| 라우팅 전문가 수 | 512 |
| 활성 전문가 / token | 22 |
| 공유 전문가 | 1개 (intermediate_size=5376) |
| 양자화 | NVFP4 (4-bit floating point) |
| 컨텍스트 길이 | 131,072 tokens |
| Mamba SSM cache dtype | float32 |
| `num_nextn_predict_layers` | 1 |

### 소프트웨어 스택

| 항목 | 버전 |
|------|-------|
| vLLM | 0.17.2rc1.dev162 |
| FlashInfer | JIT (CUDA 13.1, compute_121a) |
| CUDA Toolkit | 13.1 |
| PyTorch | 2.10.0a0 |
| 컨테이너 | nvcr.io/nvidia/nemo:dev 기반 커스텀 빌드 |

---

## 2. 전체 시도 기록 (시간순)

### ① NGC vLLM 26.02 컨테이너로 최초 시도

**시도:** `nvcr.io/nvidia/vllm:26.02-py3` (vLLM 0.15.1 기반)로 Nemotron-3-Super-120B 실행

**결과:** 즉시 실패 — vLLM 0.15.1은 Nemotron-3-Super 모델 아키텍처(NemotronH)를 지원하지 않음. upstream vLLM은 v0.17.0에서 ModelOpt mixed precision과 NemotronH 지원 추가.

> **참고:** [vLLM v0.17.0 Release](https://github.com/vllm-project/vllm/releases/tag/v0.17.0)

**조치:** `vllm/vllm-openai:latest` (0.17.1)로 전환

---

### ①-b `drop_caches` + `gpu-memory-utilization` 조정 (통합 메모리 이슈)

유저가 `sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`로 page cache 해제 방법을 알려줌 → 이후 `drop-caches` Docker 서비스로 자동화.

---

### ①-c `gpu-memory-utilization` 조정 (통합 메모리 이슈)

**시도:** 초기에 `gpu-memory-utilization=0.9` 설정

**문제:** DGX Spark 통합 메모리에서 `cudaMemGetInfo`가 ~44 GiB만 free로 보고 (실제 128 GiB 중 OS/시스템이 나머지 사용). 0.9 × 119.7 GiB = 107.73 GiB 요청 → 44 GiB free 부족으로 크래시.

**시도한 값:** 0.9 → 0.45 → 0.7 → 0.8 → 0.85 → 최종 0.8

**근본 해결:** `drop-caches` 서비스로 시작 전 page cache 해제 → free 메모리 41→100+ GiB 확보 → 0.8로 안정

---

### ② NVFP4 GEMM 백엔드 탐색

vLLM의 FP4 GEMM 경로를 하나씩 시도:

| 시도 | 백엔드 | 결과 | 비고 |
|------|--------|------|------|
| 1 | `FLASHINFER_CUTLASS` | ❌ `Error Internal` | SM121용 CUTLASS 커널이 SMEM 초과 |
| 2 | `FLASHINFER_CUDNN` | ❌ `No execution plans support the graph` | cuDNN FP4 execution plan이 SM121 미지원 |
| 3 | `FLASHINFER_CUTEDSL` | ❌ `masked_gemm` 옵션만 존재, SM121 아키텍처 미등록 | CuTe DSL admissible_archs에 sm_121 없음 |
| 4 | `VLLM_CUTLASS` | ❌ `Error Internal` (SM120 SMEM overflow) | [vLLM PR #21309](https://github.com/vllm-project/vllm/pull/21309) SM120 커널 포함, 하지만 SMEM 초과 |
| 5 | **`MARLIN`** | ✅ 정상 (W4A16 에뮬레이션) | **유일하게 작동**, `VLLM_TEST_FORCE_FP8_MARLIN=1` 필요 |

> **참고:** [FlashInfer #2577](https://github.com/flashinfer-ai/flashinfer/issues/2577) — SM120에서 mm_fp4 모든 백엔드 실패 보고
> **참고:** [vLLM #35566](https://github.com/vllm-project/vllm/issues/35566) — SM120 CUTLASS FP4 MoE garbage 출력 보고
> **참고:** [SGLang #18954](https://github.com/sgl-project/sglang/issues/18954) — SM120 NVFP4 NaN 출력 보고

---

### ③ Attention 백엔드 탐색

| 시도 | 백엔드 | 결과 | 비고 |
|------|--------|------|------|
| 1 | `FLASHINFER` (AOT) | ❌ `no kernel image` | AOT 빌드가 SM90 전용 cubin만 포함 |
| 2 | `TRITON_ATTN` | ✅ 작동 | SM121 지원, 느림 |
| 3 | `FLASH_ATTN` | ❌ SM121 미지원 | flash-attn 패키지 자체가 미지원 |
| 4 | **`FLASHINFER` (JIT)** | ✅ 작동, 빠름 | `FLASHINFER_CUDA_ARCH_LIST=12.1a`로 JIT 컴파일 |

> **핵심:** FlashInfer AOT cubin을 삭제하고 JIT로 SM121용 컴파일 강제 필요

---

### ③-b NaN 출력 디버깅

**증상:** 모델이 토큰을 생성(2.5 tok/s)하지만 텍스트 비어있음. logprobs 요청 시 NaN 에러.

**원인:** 초기 FlashInfer CUTLASS FP4 경로에서 NaN/garbage logit이 생성되어 디토크나이저가 빈 문자열 반환. Marlin으로 전환 후 해결.

**이후 재발:** FlashInfer CUTLASS FP4로 다시 전환할 때마다 빈 출력 재발 (⑪, ⑫, ⑭, ⑮ 참조). `reasoning: ""`, `content: null` 형태. 512 토큰 생성되지만 전부 빈 문자열 — FP4 GEMM 출력이 all-zeros 또는 NaN.

---

### ④ `--enforce-eager` 모드에서 Marlin으로 최초 작동 — **2.5 tok/s**

- Marlin FP4 + TRITON_ATTN + `--enforce-eager` (CUDA graph/torch.compile 비활성화)
- 모델 정상 로딩, 추론 정상, 출력 정확
- 하지만 **2.5 tok/s**로 극도로 느림

---

### ⑤ torch.compile + CUDA graph 활성화 — **14.9 tok/s** (6배 향상)

**시도:** `--enforce-eager` 제거 → torch.compile + CUDA graph (FULL_AND_PIECEWISE)

**중간 측정:**
- 첫 요청: 1.9 tok/s (torch.compile JIT 오버헤드, ~710초 컴파일)
- 두 번째 요청: **9.5 tok/s** (워밍업 중)
- 안정 상태: **14.9 tok/s** ✅

**torch.compile 캐시:** 첫 실행 시 ~710초 컴파일 후 `compile_cache`에 저장. 이후 재시작 시 `Directly load the compiled graph`로 즉시 로딩.

**결과:** 2.5 → **14.9 tok/s** ✅

---

### ⑤-b Chunked Prefill 활성화 + Prefix Caching + 메모리 프로파일링 패치로 재시작 — **13.0 tok/s**

> `--enable-chunked-prefill` 추가 (긴 프롬프트를 청크 단위로 처리하여 메모리 피크 감소)

**시도:** `--enable-prefix-caching` 추가 + 메모리 프로파일링 assertion 패치 적용 후 재시작

**측정:**
- **13.0 tok/s** (이전 14.9에서 하락)
- 12.6~13.0 tok/s로 안정적

**이유:** prefix caching 활성화로 KV cache 관리 오버헤드가 약간 증가했으나, 이후 FlashInfer JIT 전환으로 15.0 tok/s로 복구됨

**비어있는 content/reasoning 출력 문제:** 이 시점에서 응답의 `reasoning`과 `content`가 정상 출력되는 것을 첫 확인 — `"reasoning": "User wants a short greeting in Korean..."`, `"content": "안녕하세요!"`

---

### ⑥ TRITON_ATTN → FlashInfer JIT 전환 — **15.0 tok/s**

**시도:** FlashInfer AOT .so 삭제 후 JIT sm_121a 컴파일

**결과:** 14.9 → **15.0 tok/s** (TRITON 대비 ~3% 향상)

---

### ⑦ Marlin 제거 + FlashInfer CUTLASS FP4 전환 시도

**시도:** Marlin 환경변수 전부 제거, `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`로 auto-select

**발생 문제들:**

1. **`scaled_fp4_quant` CUDA sticky error** — vLLM의 `_C.abi3.so`에 포함된 FP4 quantization 커널이 SM121에서 `cudaErrorNoKernelImageForDevice`를 비동기 발생시키고, 이후 모든 CUDA 연산이 실패
   - **조치:** FlashInfer의 `nvfp4_quantize`로 대체하는 monkey-patch 적용
   - **참고:** [vLLM #36821](https://github.com/vllm-project/vllm/issues/36821) — SM121 aarch64 지원 없음

2. **`scaled_fp4_experts_quant`, `silu_and_mul_scaled_fp4_experts_quant`** — MoE 경로의 추가 C++ 커널도 동일한 sticky error
   - **조치:** 이것들도 FlashInfer로 우회 패치

3. **`cublasLt.h: No such file or directory`** — FlashInfer `nvfp4_quantize` JIT 빌드에 cuBLAS dev 헤더 필요
   - **조치:** `apt-get install libcublas-dev` 추가

---

### ⑦-b vLLM 이슈 제출 시도 (`scaled_fp4_quant` sticky error)

**시도:** `scaled_fp4_quant` CUDA sticky error에 대해 [vLLM 기존 이슈 검색](https://github.com/vllm-project/vllm/issues?q=is%3Aissue+scaled_fp4_quant+sm121+OR+sm120) 및 새 이슈 초안 작성

**제목 초안:** `[Bug]: _C.scaled_fp4_quant produces sticky CUDA error on SM121 (DGX Spark GB10), corrupts CUDA context`

**관련 이슈 발견:** [#37141](https://github.com/vllm-project/vllm/issues/37141) (Avarok upstream 요청), [#36821](https://github.com/vllm-project/vllm/issues/36821) (SM121 미지원), [#37030](https://github.com/vllm-project/vllm/issues/37030) (Marlin 토큰 오류)

**조치:** `collect_env.py` 실행으로 환경 정보 수집. 이슈 제출 준비했으나, 워크어라운드(FlashInfer `nvfp4_quantize` 대체)로 해결되어 보류.

---

### ⑧ FlashInfer CUTLASS FP4 GEMM 직접 테스트 (`mm_fp4` API)

**시도:** 컨테이너 내에서 직접 `mm_fp4()` API 호출 — vLLM 경유 없이 순수 FlashInfer 테스트

| 테스트 | 결과 |
|--------|------|
| 작은 행렬 (128x128) | ✅ 모든 6개 tactic 성공 |
| 모델 크기 (6144x6144) | ✅ 성공 |
| b.T (transpose) 전달 | ❌ shape check 실패 (`mat2.size(1) == k_packed` — 이중 transpose로 원래대로 복원되는 구조) |
| swizzled scale + mm_fp4 | 이전 sticky error로 오염된 CUDA context에서 테스트 — 재현 불가 |
| raw (no swizzle) + mm_fp4 | ✅ 성공 |
| vLLM 경유 호출 | ❌ `Error Internal` |

**원인:** vLLM profiling에서 다양한 행렬 크기를 시험할 때, SM121의 99KB SMEM을 초과하는 타일 (128x128x256B, 256x128x128B)이 선택되어 실패. 또한 `scaled_fp4_quant` sticky error가 CUDA context를 오염시켜 이후 연산이 연쇄 실패.

---

### ⑨ `nemotron_h_mtp` 메서드 시도

**시도:** `--speculative-config='{"method":"nemotron_h_mtp","num_speculative_tokens":5}'`

**결과:** vLLM이 자동으로 `method=mtp`로 전환. `nemotron_h_mtp`는 vLLM 내부에서 `mtp`로 리매핑됨 (이름만 다르고 동일 코드). 유저가 여러 차례 "왜 mtp로 바뀌냐" 질문.

> **참고:** [Nemotron Advanced Deployment Guide](https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.html#vllm) — `num_speculative_tokens > 1` 예제 포함

---

### ⑩ MTP 없이 FlashInfer CUTLASS FP4 확인 → 성공 (MTP가 OOM 원인 확인)

**시도:** MTP를 제거하고 FlashInfer CUTLASS FP4 + PIECEWISE만으로 서버 기동

**결과:** ✅ 정상 기동 — MTP drafter torch.compile이 OOM의 직접 원인 확인

---

### ⑩-b cudagraph_mode + MTP 메모리 추정 표

유저 요청으로 각 조합의 메모리 소비량을 추정:

| cudagraph_mode | MTP tokens | 모델 | compile | CUDA graph | KV cache | 총 추정 | 상태 |
|---|---|---|---|---|---|---|---|
| NONE | 0 | 70 | 0 | 0 | ~48 | ~118 | ✅ 느림 |
| PIECEWISE | 1 | 71 | 6 | 5 | ~33 | ~115 | **✅ 추천** |
| FULL+PIECE | 1 | 71 | 6 | 13 | ~25 | ~115 | ⚠️ 빠듯 |
| PIECEWISE | 3 | 71 | 10 | 8 | ~24 | ~113 | ⚠️ |
| PIECEWISE | 5 | 71 | 15 | 12 | ~15 | ~113 | ❌ OOM |

---

### ⑩-c MTP 5토큰 시도 → OOM

**시도:** `num_speculative_tokens=5` (모델의 `num_nextn_predict_layers=1`)

**결과:** OOM (모델 75GiB + MTP drafter torch.compile activation → 128GiB 초과)

> vLLM 경고: *"Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer, which may result in lower acceptance rate"*

**시도한 완화 방법:**
- `gpu-memory-utilization` 0.9 → 0.85 → 0.8 — 부족
- `cudagraph_mode=PIECEWISE` — 부족
- swap 32GiB 추가 — 여전히 OOM (128+32=160GiB 전체 소진, OOM killer 작동)
- `cudagraph_mode=NONE` — Mamba `selective_state_update` assertion error (별도 버그)
- `TORCHINDUCTOR_COMPILE_THREADS=1` — CPU 스레드만 줄이고 GPU 메모리 할당량은 미영향

**조치:** MTP 1토큰으로 축소

---

### ⑩ MTP 1 + `mamba_cache_mode=all` → 크래시

**시도:** MTP + NemotronH (Mamba hybrid) 모델

**에러:** `selective_state_update` assertion — `state_batch_indices`가 None이거나 2D가 아님

**원인:** MTP spec decode metadata의 block-level gather가 Mamba2의 `mamba_cache_mode=all` state indexing과 호환 불가

> **참고:** eugr/spark-vllm-docker PR #98에서 `--mamba-cache-mode=align` 필요성 언급

**조치:** `mamba_cache_mode=align`으로 전환 (Phase 6)

---

### ⑪ FlashInfer CUTLASS FP4 + `compute_121a` → garbage 출력, 0.3 tok/s

**시도:** 모든 `scaled_fp4_quant` sticky error를 패치한 후 FlashInfer CUTLASS FP4로 전체 모델 실행

**결과:** 0.3 tok/s + 빈 문자열 출력 (NaN/garbage logits)

> **참고:** [FlashInfer #2577](https://github.com/flashinfer-ai/flashinfer/issues/2577) — *"CUTLASS returns all zeros on SM120"* 보고

---

### ⑫ E2M1 소프트웨어 변환 패치

**시도:** SM121에 `cvt.rn.satfinite.e2m1x2.f32` PTX 명령어가 없어서, CUTLASS `float_subbyte.h`와 `quantization_utils.cuh`에서 SM121을 HW E2M1 경로에서 제외하고 소프트웨어 변환으로 대체. `quantization_utils.cuh`의 3개 함수(`float_to_e2m1`, `e2m1_to_float`, `float_to_e2m1x2`) 전부 SW fallback으로 교체.

> **참고:** [RobTand/spark-vllm-docker E2M1 patch](https://raw.githubusercontent.com/RobTand/spark-vllm-docker/flashinfer-pr-patching/e2m1_nvfp4_sm121.patch) — 동일 문제의 커뮤니티 패치

**결과:** 0.2 tok/s + 빈 문자열 — E2M1이 문제가 아니었음 (0.3→0.2로 오히려 미세 악화)

**빈 출력 상세:** 512 토큰 생성되지만 `reasoning: ""`, `content: null`. 모든 토큰이 빈 문자열 — FP4 GEMM 출력이 NaN/all-zeros로 logit이 전부 같은 값 → 항상 같은 토큰 → 디토크나이저가 빈 문자열 반환.

**별도 성공:** 이 시점에 Marlin 백엔드로 테스트 시 **14.7 tok/s**에 정상 출력 확인 (FlashInfer CUTLASS가 아닌 Marlin 경로)

---

### ⑬ `fastsafetensors` 로딩 포맷 적용 + JIT prebuild 개선

**시도:** `--load-format=fastsafetensors` (병렬 safetensors 로더) + JIT prebuild를 모델 로딩 이전에 실행하도록 변경 (entrypoint에서 `prebuild_flashinfer_jit.py` 먼저 실행)

**결과:**
- 모델 로딩 시간 **400초 → 92초** (4.3배 단축) ✅
- JIT prebuild: 최초 ~25분 (fused_moe 포함), 이후 캐시 마커 파일로 건너뛰기
- `MAX_JOBS` 제한 해제: prebuild 시 CPU 전체 코어(20)로 병렬 빌드, 모델 로딩 시에는 `MAX_JOBS=2`로 제한 (메모리 보호)

---

### ⑭ `compute_120f` 시도 (CUDA 13.1 nvcc)

**시도:** `FLASHINFER_CUDA_ARCH_LIST=12.0f`로 TMA fast-path 타겟 컴파일

> **참고:** [CUTLASS #3096](https://github.com/NVIDIA/cutlass/issues/3096) — `compute_120f`로 39 tok/s 정상 출력 보고 (SGLang)

**결과:** `cudaErrorMisalignedAddress` → TMA 명령어가 CUDA 12.9 런타임과 ABI 불일치

**조치:** CUDA 13.1 nvcc + CUDA 13.1 런타임 조합으로 재시도

---

### ⑭ `compute_120f` + CUDA 13.1 전체 → 4.5 tok/s + garbage

**시도:** CUDA 13.1 전체 환경에서 `compute_120f`

**결과:** 4.5 tok/s (0.3보다 나음) 하지만 여전히 빈 문자열 출력

**결론:** FlashInfer CUTLASS FP4 GEMM이 SM121에서 수치적으로 올바른 결과를 생성하지 못함 (upstream 버그)

---

### ⑮ GDC 플래그 추가 시도

**시도:** `FLASHINFER_EXTRA_CUDAFLAGS="-DCUTLASS_ENABLE_GDC_FOR_SM100=1"` — PDL 배리어를 실제 명령어로 컴파일

> **참고:** eugr/spark-vllm-docker의 GDC 플래그 설정

**결과:** 여전히 0.3 tok/s + garbage — GDC만으로는 해결 안 됨

---

### ⑯ `compute_120f` + SM121 → Illegal instruction

**시도:** `FLASHINFER_CUDA_ARCH_LIST=12.0f`를 SM121 하드웨어에서 실행

**결과:** `cudaErrorIllegalInstruction` — `compute_120f`는 SM120(RTX 5090) 전용, SM121에서 일부 명령어 호환 불가

> **참고:** eugr PR #98 — *"FlashInfer는 12.1a로, vLLM은 12.0a로 따로 컴파일"*

**조치:** `12.1a`로 복귀

---

### ⑰ eugr/spark-vllm-docker 이미지 시도

**시도:** [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) prebuilt 이미지 (CUTLASS 4.4.2, K=64 타일, GDC flag 내장)

| 시도 | 결과 |
|------|------|
| eugr + FlashInfer CUTLASS FP4 + MTP | ❌ `fp8_gemm_sm100` segfault (`Cannot access data pointer of Tensor that doesn't have storage`) |
| eugr + `kv-cache-dtype=fp8` | ❌ 동일 segfault |
| eugr + FP8 KV cache 비활성화 | ❌ FlashInfer FP4 GEMM에서도 동일 `Cannot access data pointer` |
| eugr + Marlin + FlashInfer attention + PIECEWISE | ❌ FlashInfer TVM FFI segfault |
| eugr + Marlin + `cudagraph_mode=NONE` | ❌ 서버 뜨지만 첫 요청에서 동일 TVM FFI 크래시 |

> **원인:** eugr 이미지의 FlashInfer (main 브랜치 빌드)에서 `torch.compile` + TVM FFI 비호환 문제. 이전 커스텀 Dockerfile의 FlashInfer 0.6.6 pip 버전에서는 발생 안 했음

**조치:** eugr 이미지 포기, 기존 커스텀 Dockerfile로 복귀

---

### ⑱ TMA WS Grouped GEMM 80 Tactics Skip 문제 분석

**증상:** 커스텀 Dockerfile로 서버 기동 성공 후, autotuner 로그에서:
```
flashinfer.jit: [Autotuner]: Skipping tactic <MoERunner> 14, due to failure:
[TensorRT-LLM][ERROR] Assertion failed: Failed to initialize cutlass TMA WS grouped gemm.
Error: Error Internal (cutlass_kernel_file_gemm_grouped_sm120_M128_BS_group2.generated.cu:60)
```

**규모:** MoE autotuner가 시도하는 전체 tactic 중 **80개** TMA Warp Specialized tactics가 전부 skip되어 느린 fallback tactics만 사용됨.

**원인 분석:**
1. `StageCountAutoCarveout`가 SM121의 101KB SMEM에서 3 pipeline stages를 계산
2. 실제로는 `alignas(1024)` 패딩 + `SharedStorage::TensorStorage` overhead로 2 stages만 가능
3. 3 stages로 인스턴스화 → SMEM overflow → runtime initialization failure

> **참고:** [CUTLASS #2820](https://github.com/NVIDIA/cutlass/issues/2820) — SM120 Block-Scaled MMA Runtime Assertion Failure

---

### ⑲ TRT-LLM PR #5823 적용 시도 (C++ `== 100` → `>= 100`)

**시도:** [TensorRT-LLM PR #5823](https://github.com/NVIDIA/TensorRT-LLM/pull/5823) — `moe_gemm_tma_ws_launcher.inl` line 390의 `ArchTag::kMinComputeCapability == 100` → `>= 100`

**결과:** SM121이 SM100 코드 경로를 타면서 `static assertion failed: Specialization requires Stages set to value 2 or more`

**원인:** SM121(101KB SMEM)에서 `StageCountAutoCarveout`가 1 stage만 가능하다고 판단 — SM100(228KB)보다 SMEM이 절반도 안 되기 때문

> **참고:** [TRT-LLM PR #11997](https://github.com/NVIDIA/TensorRT-LLM/pull/11997), [PR #12309](https://github.com/NVIDIA/TensorRT-LLM/pull/12309) — SM120/SM121 Python-side 가드 해제

---

### ⑲ StageCount<2> 강제 + SMEM 오버플로우 MoE 타일 제거

**시도:** SM120에서 `StageCountAutoCarveout` 대신 `StageCount<2>` 강제

**결과:** 128x128x256, 128x256x128 타일이 `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)`에서 SMEM 초과로 `Error Internal`

**조치:** heuristic, dispatcher, code generator에서 SMEM 초과 타일 제거, 128x128x128만 남김

---

### ⑳ CUTLASS commit 73c59c0 (`ReducedSmemCapacityBytes`) 적용

**시도:** [CUTLASS commit 73c59c0](https://github.com/NVIDIA/cutlass/commit/73c59c0) — `sm120_blockscaled_mma_builder.inl`에서 grouped GEMM overhead (scheduler pipeline, tensor map, CLC response)를 빼고 `ReducedSmemCapacityBytes`로 stage 수 계산

**결과:** 여전히 stage < 2 — SM121의 101KB에서 overhead를 빼면 128x128x128 타일도 2 stage 안 맞음

> **핵심 계산:** `sm120_smem_capacity_bytes = 101376` (99KB). Grouped GEMM overhead ~10KB를 빼면 ~91KB. 128x128x128 BlockScaled 타일 stage당 ~50KB → 1.8 stage → 1 stage → assertion 실패

---

### ㉑ K=64 타일 (`CtaShape128x128x64B`) 추가 시도

**시도:** [FlashInfer PR #2786](https://github.com/flashinfer-ai/flashinfer/pull/2786)의 K=64 타일을 `are_tile_shapes_supported_sm120()`에 추가

> **참고:** [FlashInfer PR #2786](https://github.com/flashinfer-ai/flashinfer/pull/2786) — K=64 타일 추가로 SM120 MoE 성능 2배 향상

**결과:** `undefined symbol: ...tma_warp_specialized_generic_moe_gemm_kernelLauncher<...Shape<128,128,64>...>` — K=64 + SM120 + BlockScaled 조합의 커널 인스턴스화 코드가 실제로 구현되어 있지 않음 (선언만 있고 정의 없음)

**조치:** K=64 제거, K=128만으로 진행

---

### ㉒ MTP Drafter의 Unquantized MoE (BF16 TRITON) 발견

**발견:** 서버 로그에서 `Using TRITON backend for Unquantized MoE` 경고. 유저 질문: "이건 왜 뜨는거임?"

**설명:** Nemotron-3-Super의 MTP head에 MoE 레이어가 있으나, 이 MoE는 메인 모델과 달리 **양자화되지 않은 BF16 weight**. BF16 MoE에는 FlashInfer CUTLASS FP4가 아닌 Triton 커널이 사용됨. 정상 동작.

---

### ㉒-b RTX PRO 6000 SM120 벤치마크 데이터 공유

**유저 공유:** 4× RTX PRO 6000 (SM120, 96GB GDDR7)에서의 상세 벤치마크 보고서:
- Marlin + TP4 + MTP 없음: **50.5 tok/s** (best)
- Marlin + MTP=2: **39.6 tok/s** (-22% — MTP가 Marlin에서 오히려 성능 저하)
- FlashInfer CUTLASS + MTP=2: **40-41 tok/s** (80 TMA WS tactics skipped)
- **핵심 발견:** MTP + Marlin은 성능 저하 (W4A16 디퀀타이제이션 ≠ 네이티브 FP4 → acceptance rate 61-85%)

이 데이터가 DGX Spark에서도 Marlin 대신 네이티브 FlashInfer CUTLASS FP4 추구의 근거가 됨.

---

### ㉓ 128x128x128 타일만으로 FlashInfer CUTLASS FP4 MoE 성공

**시도:** `CtaShape128x128x64B` (128x128x128 FP4 elements)만 남기고 나머지 전부 제거

**결과:**
- ✅ JIT 빌드 성공 (fused_moe_120.so 컴파일)
- ✅ Autotuner 정상 완료 (에러 없음, 3초)
- ✅ CUDA 그래프 캡처 성공 (51/51)
- ✅ 서버 정상 기동

**새 문제:** 추론 시 `cudaErrorIllegalInstruction` — Mamba mixer2 (layer 44)에서 크래시. MoE가 아닌 SSM 레이어 문제. vLLM `_C.abi3.so`의 Mamba 커널이 SM121에서 호환 안 됨

---

### ㉓ vLLM + FlashInfer 소스 빌드 시도

**시도:** vLLM과 FlashInfer를 소스에서 빌드하여 SM121용 `_C.abi3.so` 생성

**발생 문제:**
- `_C_stable_libtorch` 컴파일 에러 (vLLM PR #34758, #34302 revert 충돌)
- FlashInfer JIT 빌드 중 다수 에러 (`MAX_JOBS` 관련, ninja 빌드 실패)
- 여러 차례 빌드 실패 후 재시도

> **참고:** [vLLM PR #34758](https://github.com/vllm-project/vllm/pull/34758) — 빌드 충돌 관련

**결과:** 결국 성공하여 SM121 네이티브 `_C.abi3.so` 확보. 하지만 Mamba `cudaErrorIllegalInstruction` 문제가 별도로 남음 (vllm#37431).

---

### ㉔ Triton Mamba async 버그 해결 (`mamba_mixer2_patch.py`)

**시도:** vllm#37431에서 보고된 SM12x Triton PTX codegen의 메모리 배리어 누락 문제

**증상:** `cudaErrorIllegalInstruction` / `cudaErrorMisalignedAddress` — Mamba mixer2 레이어에서 비동기 크래시

**수정:** `mamba_mixer2_patch.py` — Mamba 레이어 전후 `torch.cuda.synchronize()` 삽입 (CUDA graph 캡처 중에는 건너뛰기)

> **참고:** [vLLM #37431](https://github.com/vllm-project/vllm/issues/37431) — SM12x Triton Mamba async bug

**결과:** ✅ Mamba 크래시 해결

---

### ㉕ 커스텀 Dockerfile 복귀 + 전체 재구성 → 성공 🎉

**결론:** FlashInfer CUTLASS FP4 GEMM은 SM121에서 production-ready가 아님. eugr 커뮤니티도 Nemotron에 Marlin 사용.

**최종 구성:**
- **Linear FP4:** FlashInfer CUTLASS (JIT sm_121a)
- **MoE FP4:** FlashInfer CUTLASS (패치된 heuristic + StageCount<2>)
- **Attention:** FlashInfer (JIT sm_121a)
- **MTP:** 활성화 (`mamba_cache_mode=align`)
- **CUDA graph:** `full_and_piecewise`

이 조합으로 커스텀 Dockerfile + 9개 패치 파일로 안정적 서빙 달성

---

### ㉖ 동시 요청 배칭 테스트 — MoE amortization 발견

**시도:** 단일 요청 ~19.8 tok/s에서 MoE expert 읽기가 배치 내에서 amortize되는지 확인

| 동시 요청 수 | Aggregate throughput | Drafted/s | Per-user | 배율 |
|-------------|---------------------|-----------|----------|------|
| 1 | 19.8 tok/s | 11.1 | 19.8 | 1.0× |
| 2 | **31.7 tok/s** | 18.0 | 15.9 | **1.60×** |
| 4 | **53.4 tok/s** | 28.4 | 13.4 | **2.69×** |
| 5 | **62.7 tok/s** (서버 로그) | — | 12.5 | **3.17×** |

**서버 엔진 메트릭 (5 concurrent):**
- Engine generation throughput: **52.7 tok/s** (서버 로그, 피크 62.7)
- 4 concurrent MTP counting: Engine **145.8 tok/s** (thinking tokens 포함), wall-clock per-user **38.3 tok/s**

**핵심 발견:** MoE 전문가 읽기가 배치에서 공유되어 aggregate throughput이 거의 선형 증가. 단일 요청의 병목인 scatter bandwidth가 동시 요청에서 amortize.

---

### ㉗ MTP=2로 증가 — **~22 tok/s**

**변경:**
```
--speculative-config={"method":"mtp","num_speculative_tokens":2}  # 1 → 2
--max-num-batched-tokens=8400  # 8352 → 8400 (MTP=2 블록 크기 요구)
```

**발생 문제:** `AssertionError: block_size (8400) must be <= max_num_batched_tokens (8352)` → `max_num_batched_tokens=8400`으로 해결

**세부 측정 (MTP=1 vs MTP=2):**

| 테스트 유형 | MTP=1 | MTP=2 | 개선 |
|------------|-------|-------|------|
| Counting (반복적, 높은 acceptance) | 21.8 tok/s | **27.0 tok/s** | **+24%** |
| Essay (창의적) | 19-20 tok/s | 20.7 tok/s | +5% |
| Code generation | 19 tok/s | ~25 tok/s | +32% |

**MTP=2 서버 메트릭:**
- Mean acceptance length: 2.33~2.84 (max 3.0)
- Per-position acceptance rate: [79-97%, 53-88%]
- Forward pass rate (drafted throughput): **~11.2 fwd/s** (90ms per forward, MoE scatter bandwidth 제한)
- Drafted token throughput: ~18 tok/s (MTP=2이면 11.2 × (1+accepted) ≈ 18)

**CUDA_LAUNCH_BLOCKING=1 테스트:** Mamba Triton async 버그 워크어라운드로 시도 시 14→8.8 tok/s (**37% 성능 저하**). `torch.cuda.synchronize()` 워크어라운드가 더 효율적.

**결과:** 19.8 → **~22 tok/s** (+12%) — MTP acceptance rate [79-97%, 53-88%]

---

### ㉔ Marlin에서 Marlin 없이 전환 과정

사실 최종 구성에서 Marlin은 **사용하지 않습니다**. 이전 시도들에서:
- Marlin + FlashInfer Attention: **14.6 tok/s** (Marlin MoE 포함)
- 하지만 MTP acceptance rate가 Marlin에서 낮음 (W4A16 디퀀타이제이션이 네이티브 FP4와 수치 다름)

> **참고:** RTX PRO 6000 SM120 벤치마크에서 Marlin + MTP = **-22% 성능 저하** (acceptance rate 61-85%)

최종적으로 FlashInfer CUTLASS FP4를 패치하여 MoE까지 네이티브 FP4로 전환 → MTP와 수치적 일치 → 높은 acceptance rate

---

## 3. 시도했으나 효과 없거나 실패한 것들 (최종 세션)

### ❌ FlashInfer 0.6.6 pip 업그레이드 (FlashInfer CUTLASS FP4 수정 시도)

**시도:** FlashInfer 0.6.5 → 0.6.6으로 pip 업그레이드. [FlashInfer PR #2670](https://github.com/flashinfer-ai/flashinfer/pull/2670) (SM120 SMEM fix, `StageCount<2>`)이 포함될 것으로 기대.

**결과:** 동일한 `Error Internal` — PR #2670의 수정이 pip 0.6.6 릴리스에 포함되지 않았거나 MoE grouped GEMM 경로에는 적용되지 않음. SM121 인식만 추가됨 ([PR #2631](https://github.com/flashinfer-ai/flashinfer/pull/2631)).

---

### ❌ swizzle_blockscale / `is_sf_swizzled` 테스트

**시도:** `scaled_fp4_quant(x, gsf, True)` (swizzled=True) vs `scaled_fp4_quant(x, gsf, False)` (raw) 비교 테스트

**결과:** 둘 다 후속 연산에서 CUDA sticky error 발생. swizzle 여부와 무관하게 vLLM `_C.abi3.so`의 FP4 quantization 커널 자체가 SM121에서 sticky CUDA error를 발생시킴.

---

### ❌ `flashinfer-cudnn` 백엔드 (cuBLASLt FP4)

**시도:** CUTLASS SMEM 문제를 우회하기 위해 cuDNN/cuBLASLt 경유 FP4 GEMM

**에러:** `cudnn._compiled_module.cudnnGraphNotSupportedError: No execution plans support the graph`

**원인:** cuDNN의 FP4 execution plan이 SM121을 지원하지 않음

---

### ❌ `compile_sizes=[]` (torch.compile 크기 지정 비활성화)

**시도:** MTP drafter의 torch.compile OOM을 회피하기 위해 `compile_sizes=[]`로 warmup 컴파일 건너뛰기

**결과:** 메모리 절약 효과 있었으나 `cudagraph_mode=NONE`에서 Mamba `selective_state_update` assertion error가 별도로 발생하여 무관하게 실패

---

### ❌ BF16 → FP16 activation casting / FP32 scale precision

**시도:**
- `flashinfer#2577`에서 제보된 "scale factor를 `.float()`로 변환하면 해결" 적용
- BF16 activation을 FP16으로 변환하여 CUTLASS FP4 GEMM에 전달

**결과:** FlashInfer CUTLASS FP4 GEMM 출력 여전히 garbage (0.3 tok/s, 빈 문자열)

---

### ❌ vLLM 메모리 프로파일링 패치 (2건)

**시도:** DGX Spark 통합 메모리에서 `cudaMemGetInfo`가 보고하는 free 메모리가 부정확하여:
- `vllm/utils.py` startup memory check 패치 (free < requested 체크 우회)
- `vllm/worker/gpu_worker.py` profiling assertion 패치 (`init_free < post_profile_free` 체크 우회)

**결과:** 작동하지만 이후 `drop-caches` 서비스로 근본 해결하여 패치 제거

---

### ❌ causal-conv1d / mamba-ssm 네이티브 CUDA 커널 시도

**시도:** Triton Mamba 커널의 SM121 async 버그를 회피하기 위해 [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)와 [mamba-ssm](https://github.com/state-spaces/mamba) 네이티브 CUDA 커널로 교체

> **참고:** [vLLM #37431](https://github.com/vllm-project/vllm/issues/37431) — 유저가 "이 문제랑 같은 문제 아냐?" 라며 공유

**결과:** NGC 26.01 PyTorch와 확장 호환 문제로 빌드 실패. 대신 `torch.cuda.synchronize()` 워크어라운드 적용 (`mamba_mixer2_patch.py`)

---

### ❌ NVIDIA 드라이버 595.71 불안정 / 시스템 재부팅

**증상:** 여러 차례 OOM, swap 소진 후 `nvidia-smi`에서 GPU 인식 실패

**시도:** `nvidia-smi` 확인 → 드라이버 상태 이상 → 시스템 재부팅

**결과:** 재부팅 후 정상 복구. 이후 `drop-caches` 서비스 + `gpu-memory-utilization=0.8`로 안정화

---

### ❌ `VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=0` (FP8 ScaledMM 차단)

**시도:** FlashInfer FP8 ScaledMM 경로가 SM121에서 segfault를 발생시켜 환경변수로 차단

**결과:** 작동. 하지만 이후 `patch_fp8_sm120.py`에서 코드 레벨로 SM120+ 차단하여 환경변수 불필요

---

### ❌ `restart: unless-stopped` 설정

**시도:** 서버 크래시 시 자동 재시작을 위해 Docker restart policy 설정

**결과:** 통합 메모리 환경에서 OOM 크래시 → 재시작 → 재OOM 무한 루프 위험. `restart: none`으로 최종 결정

---

### ❌ `VLLM_COMPILE` 환경변수 (torch.compile 모드)

**시도:** `VLLM_COMPILE=1` 환경변수로 torch.compile 활성화 (초기 방식)

**결과:** 이후 `--compilation-config` CLI 인자로 전환하여 더 세밀한 제어 가능. `VLLM_COMPILE` 환경변수는 제거.

---

### ❌ `VLLM_SKIP_GPU_MEM_CHECK=1` (메모리 체크 우회)

**시도:** DGX Spark 통합 메모리에서 `cudaMemGetInfo` free 메모리가 부정확하여 startup memory check를 환경변수로 우회

**결과:** 작동하지만 `drop-caches` 서비스로 page cache를 해제하면 정확한 free 메모리가 보고됨 → 환경변수 불필요. Dockerfile의 `utils.py` + `gpu_worker.py` 메모리 패치와 함께 제거.

---

### ❌ `torch.set_float32_matmul_precision('high')` 경고

**증상:** torch.compile 시 `TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled` 경고

**시도:** 해당 설정 적용 고려

**결과:** FP4 모델에서는 float32 matmul이 거의 없어 효과 없음. 경고만 무시.

---

### 기타 Docker 설정 (성능 무관, 안정성 목적)

| 설정 | 값 | 이유 |
|------|-----|------|
| `ipc: host` | 호스트 IPC 공유 | CUDA IPC, NCCL 통신 |
| `shm_size: "16g"` | 공유 메모리 16GB | PyTorch DataLoader, NCCL 버퍼 |
| `ulimits: memlock: -1` | 메모리 락 제한 해제 | GPU 메모리 핀 |
| `--enable-sleep-mode` | GPU 유휴 시 메모리 해제 | 통합 메모리 절약 |
| `--tool-call-parser=qwen3_coder` | 툴 콜 파서 | Nemotron-3-Super 호환 |
| `--reasoning-parser=super_v3` | reasoning 파서 | [super_v3_reasoning_parser.py](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) |
| `docker compose build` | 커스텀 Dockerfile 빌드 | 패치 적용 이미지 생성 (~6분) |

---

### ❌ `fuse_attn_quant: true` — Attention-Quantization 퓨전

**시도:**
```
"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true,"fuse_attn_quant":true}
```

**에러:** `AssertionError: Query must be FP8 when attn+quant fusion happened`

**원인:** `fuse_attn_quant`은 FP8 query를 기대하지만, SM120 XQA 호환성을 위해 `disable_flashinfer_q_quantization: true`가 필수. BF16 query + fuse_attn_quant은 근본적으로 호환 불가.

**부작용:** CUDA graph 모드가 `FULL_AND_PIECEWISE` → `FULL_DECODE_ONLY`로 다운그레이드됨

**조치:** `fuse_attn_quant` 제거

---

### ❌ `max_autotune: true` — GEMM 자동 튜닝

**시도:** `"pass_config":{"max_autotune":true,...}`

**에러:** `Not enough SMs to use max_autotune_gemm mode`

**원인:** GB10의 48 SM이 GEMM autotuning threshold 미만
**부작용:** 컴파일 시간 증가 (55→91초), Mamba cache blocks 감소 (147→117)

**조치:** 즉시 되돌림

---

### ❌ THP (Transparent Huge Pages) 활성화

**시도:** privileged 컨테이너에서 THP + defrag 활성화 (sudo 불가로 docker 서비스 이용)
```sh
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo always > /sys/kernel/mm/transparent_hugepage/defrag
```

**결과:** GPU 성능에 측정 가능한 효과 없음. 통합 메모리 아키텍처에서 THP 무의미.

---

### ❌ MTP=3 — 투기적 토큰 3개

**시도:**
```
--speculative-config={"method":"mtp","num_speculative_tokens":3}
--max-num-batched-tokens=8448
```

**발생 문제:** `AssertionError: block_size (8432) must be <= max_num_batched_tokens (8400)` → 8448로 해결

**결과:** MTP=2 대비 동일하거나 오히려 악화
- 3번째 위치 acceptance rate: 48-59% (너무 낮아 overhead > 이점)
- 이 모델은 `num_nextn_predict_layers=1`이라 같은 MTP 레이어를 3번 반복 → 점점 부정확

**조치:** MTP=2로 되돌림

---

### ❌ MTP=5 — 투기적 토큰 5개

**시도 (초기 세션):** `num_speculative_tokens=5`

**결과:** OOM — 모델 75GiB + MTP drafter 5회 forward activation → 128GiB 통합 메모리 초과

**시도한 완화:**
- swap 32GiB 추가 → 여전히 OOM (128+32=160GiB 전체 소진)
- `gpu-memory-utilization` 0.9→0.85→0.8 → 부족
- `TORCHINDUCTOR_COMPILE_THREADS=1` → CPU만 줄임, GPU 메모리 무영향

**조치:** MTP=1로 축소 후 나중에 MTP=2로 최종 결정

---

### ❌ nsys GPU 프로파일링 (4가지 시도)

**시도 1:** `nsys profile ... python3 -m vllm.entrypoints.openai.api_server`
- 결과: "SKIPPED: does not contain CUDA kernel data" — CUDA graph replay가 분해되지 않음

**시도 2:** `--cuda-graph-trace=node` 추가 + delay 540초 (서버 로딩 대기)
- 결과: CUPTI 이벤트 2079개 수집. `"Hardware tracing used for CUDA tracing"` — 커널 activity 테이블 미생성
- SM121 하드웨어 트레이서의 한계로 판단

**시도 3:** `--cuda-trace-scope=system-wide` (docker exec에서)
- 결과: 컨테이너 격리로 CUDA trace data 미수집

**시도 4:** `torch.profiler` (클라이언트 프로세스에서 CUDA context 초기화 후)
- 결과: `cudaDeviceSynchronize` 650us만 캡처 — 서버의 GPU 커널은 보이지 않음 (CUPTI per-process)

**시도 5:** `nsys profile --attach-process=157` (EngineCore PID)
- 결과: `unrecognised option` — nsys 2025.6.1에 해당 옵션 없음

**결론:** SM121에서 nsys GPU 커널 프로파일링 현재 불가능

---

### ❌ `--kv-cache-dtype=fp8` (eugr 레시피)

**시도:** eugr `vllm-node` 이미지에서 FP8 KV 캐시

**결과:** `Cannot access data pointer of Tensor that doesn't have storage` — FlashInfer `bmm_fp8` → `fp8_gemm_sm100` segfault

**원인:** FlashInfer FP8 ScaledMM이 SM121에서 불안정

---

### ❌ `cudagraph_mode=NONE` + MTP → Mamba assertion

**에러:** `selective_state_update` assertion — cudagraph_mode와 무관하게 MTP + NemotronH 조합 버그

**조치:** `mamba_cache_mode=align`으로 해결 (Phase 6)

---

## 4. 성능 변화 추이

| Phase | 커밋 | 주요 변경 | tok/s (단일) | 변화 |
|-------|-------|-----------|-------------|------|
| 최초 시도 | — | NGC 26.02 (vLLM 0.15.1) | 실패 | — |
| Marlin + enforce-eager | — | TRITON_ATTN, no compile | 2.5 | — |
| +torch.compile (첫 warmup) | — | JIT 오버헤드 | 1.9 → 9.5 | 워밍업 중 |
| +torch.compile + CUDA graph | — | FULL_AND_PIECEWISE | **14.9** | **+496%** |
| +prefix caching | — | KV cache 재사용 | 13.0 | -13% (오버헤드) |
| +FlashInfer JIT Attention | — | TRITON → FLASHINFER | **15.0** | +15% |
| FlashInfer CUTLASS FP4 시도 | — | compute_121a | 0.3 | garbage 출력 |
| +E2M1 패치 | — | SW E2M1 변환 | 0.2 | 여전히 garbage |
| +compute_120f (CUDA 13.1) | — | TMA fast-path | 4.5 | 여전히 garbage |
| Marlin MoE (standalone) | — | 모든 Marlin | 14.6 | 정상 출력 |
| +E2M1 패치 후 Marlin 테스트 | — | Marlin + FlashInfer Attn | 14.7 | 정상 출력 |
| 0 | `5b1fb7e` | 초기 구축 (커스텀 Dockerfile) | 실패 | — |
| 1 | `c242f9b` | FlashInfer CUTLASS FP4 + MTP 시도 | 0.3 | garbage |
| 2 | `8ad7a15` | SM121 런타임 패치 3종 | **~12.0** | **Baseline** |
| — | — | (첫 warmup 1.2, 이후 12.3~12.4) | 12.3 | — |
| 3 | `dd3a3e3` | torch.compile (no CUDA graph) | ~12.4 | +3% |
| — | — | +fuse_norm_quant (피크 12.8, 안정 12.6) | 12.6 | — |
| 4 | `3b561e7` | fuse_norm_quant + fuse_act_quant | ~12.8 | +3% |
| 5 | `d70b5bf` | MTP speculative (tokens=1) | **~17.1** | **+33%** ⭐ |
| — | — | (MTP 100 tok 5.8초=17.2, 200 tok 13초=15.4) | 15.4~17.1 | — |
| 6 | `70b814c` | Prefix caching + mamba_cache_mode=align | ~17.1 | — |
| — | — | (피크 15.8, 평균 17.1) | 15.8~17.1 | — |
| 7 | `be637ce` | CUDA graph PIECEWISE | ~17.9 | +5% |
| 8 | `a412be1` | SM121 CUDA graph 패치 3종 (FULL_AND_PIECEWISE) | **~19.8** | +11% |
| — | — | (피크 21.8 counting, 19.5 일반) | 19.5~21.8 | — |
| — | — | +CUDA_LAUNCH_BLOCKING=1 (Mamba workaround) | 8.8 | -56% (비채택) |
| 9 | 미커밋 | MTP=2 (counting 27.0, essay 20.7) | **~22.1** | +12% |
| — | — | **Baseline (12.0) 대비 총 개선** | — | **+84%** |
| — | — | **최초 (2.5) 대비 총 개선** | — | **+784%** |

---

## 5. 벤치마크 결과

> 벤치마크 실행 일시: 2026-03-22
> vLLM 버전: 0.17.2rc1.dev162
> 벤치마크 도구: `vllm bench serve` (공식 CLI, [문서](https://docs.vllm.ai/en/latest/benchmarking/cli/#online-benchmark))
> 데이터셋: `--dataset-name random --input-len 128 --output-len 256`
> 측정 방법: rate=1 (10 prompts, 단일 사용자 시뮬레이션) + rate=inf (20 prompts, 최대 처리량)

### 5-0. 설정 조합별 벤치마크 (12가지)

각 설정 조합마다 서버를 재시작하고 `vllm bench serve`로 측정한 결과입니다.

| # | 설정 | Attention | Out tok/s (rate=1) | Out tok/s (rate=inf) | TPOT (ms) | P99 ITL (ms) | MTP Accept |
|---|------|-----------|-------------------:|---------------------:|----------:|-------------:|-----------:|
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

**핵심 분석:**

- **Random dataset에서 MTP는 역효과**: acceptance rate 15-36%로, 예측 가능한 패턴이 없어 MTP overhead가 이점을 초과합니다. 실제 텍스트에서는 66-92% acceptance로 MTP=2가 최적입니다 (아래 5-1 참조).
- **CUDA graph 효과**: MTP=0에서 piecewise(51.95) vs none(52.84) vs enforce-eager(54.54) — random dataset에서는 큰 차이 없음. 하지만 실제 사용에서 CUDA graph는 latency jitter를 줄여줍니다.
- **Fusion 효과**: MTP=2 기준 fusion ON(32.86) vs OFF(34.19) — random에서는 미미하지만 TPOT가 8ms 차이 (146→154ms).
- **Attention**: FLASHINFER(32.86) vs TRITON_ATTN(38.76) — TRITON이 random에서 약간 빠르지만, 실제 텍스트에서는 FlashInfer가 더 좋은 MTP acceptance를 보입니다.
- **최대 처리량 (rate=inf)**: MTP=1+piecewise가 **101.66 tok/s**로 가장 높음 (random dataset 기준).

---

### 5-1. 실제 텍스트 — 단일 요청 (Single User Experience)

5개 서로 다른 프로그래밍/기술 프롬프트, `max_tokens=512`, `temperature=0.7`

| 요청 | 출력 토큰 | 소요 시간 | 처리량 |
|------|----------|----------|--------|
| Python quicksort 구현 | 512 | 21.7s | **23.6 tok/s** |
| 상대성이론 설명 | 512 | 26.2s | **19.6 tok/s** |
| Docker 가이드 | 512 | 24.9s | **20.5 tok/s** |
| BST 구현 | 512 | 20.6s | **24.8 tok/s** |
| 컴퓨팅 역사 | 512 | 23.2s | **22.0 tok/s** |

| 메트릭 | 값 |
|--------|-----|
| **평균 처리량** | **22.1 tok/s** |
| 최소 | 19.6 tok/s |
| 최대 | 24.8 tok/s |

**서버 로그 MTP 메트릭 (단일 요청 시):**
- Mean acceptance length: 2.33~2.84
- Per-position acceptance rate: [79-97%, 53-88%]
- Draft acceptance rate: 66-92%

### 5-2. 실제 텍스트 — 5개 동시 요청 (Multi-User Throughput)

5개 서로 다른 프롬프트 동시 전송, `max_tokens=512`

| 요청 | 출력 토큰 | 소요 시간 | 개별 처리량 |
|------|----------|----------|------------|
| Web scraper 구현 | 512 | 49.2s | 10.4 tok/s |
| 양자 컴퓨팅 설명 | 512 | 49.2s | 10.4 tok/s |
| FastAPI REST API | 512 | 46.1s | 11.1 tok/s |
| ML 파이프라인 설명 | 512 | 51.5s | 9.9 tok/s |
| SQL vs NoSQL 비교 | 512 | 50.7s | 10.1 tok/s |

| 메트릭 | 값 |
|--------|-----|
| **Aggregate 처리량** | **49.7 tok/s** |
| **유저당 평균** | **10.4 tok/s** |

### 5-3. vllm bench serve — Random Dataset

`--dataset-name random --input-len 128 --output-len 256`

#### Request Rate = 1.0 (20 prompts)

| 메트릭 | 값 |
|--------|-----|
| Output token throughput | 38.05 tok/s |
| Peak output throughput | 42.00 tok/s |
| Total token throughput | 57.07 tok/s |
| Mean TPOT | 154.88 ms |
| Mean ITL | 206.78 ms |
| P99 ITL | 374.14 ms |
| MTP Acceptance rate | 16.88% |

#### Request Rate = inf (50 prompts, max throughput)

| 메트릭 | 값 |
|--------|-----|
| Output token throughput | **42.57 tok/s** |
| Total token throughput | **63.86 tok/s** |
| Mean TPOT | 155.23 ms |
| P99 ITL | 371.83 ms |
| MTP Acceptance rate | 18.65% |

> **참고:** Random dataset의 MTP acceptance ~17%는 예측 패턴이 없어서. 실제 사용 시 66-92%.

---

## 6. 패치 파일 목록

### 런타임 패치 (Volume Mount)

| 파일 | 줄 수 | 역할 | 마운트 대상 |
|------|------|------|-----------|
| `tvm_build_patch.py` | 918 | TVM FFI C DLPack `has_storage()` 수정 ([PyTorch #122706](https://github.com/pytorch/pytorch/issues/122706) 관련) | `tvm_ffi/utils/_build_optional_torch_c_dlpack.py` |
| `cutlass_heuristic_patch.cpp` | 769 | SM121 SMEM에 맞는 타일 크기 제한 (`CtaShape128x128x64B`만) | `flashinfer/data/.../cutlass_heuristic.cpp` |
| `prebuild_flashinfer_jit.py` | 107 | AOT `fused_moe_120.so` 삭제 + JIT 사전 빌드 + 캐시 마커 | `/workspace/prebuild_flashinfer_jit.py` |
| `vllm_flashinfer_utils_patch.py` | 796 | `supports_trtllm_attention()` SM120 지원 + `UNIFORM_BATCH` CG support 반환 (기존 `UNIFORM_SINGLE_TOKEN_DECODE`만 반환하여 FULL graph 불가) | `vllm/utils/flashinfer.py` |
| `flashinfer_backend_patch.py` | 1,818 | `get_required_kv_cache_layout()` SM120 HND 반환 + XQA spec decode 시 `q_seq_len > 1`일 때 causal attention mask 생성 | `vllm/v1/attention/backends/flashinfer.py` |
| `mamba_mixer2_patch.py` | 956 | `conv_ssm_forward` 전후 `torch.cuda.synchronize()` (SM121 Triton PTX codegen 메모리 배리어 누락 워크어라운드, [vllm#37431](https://github.com/vllm-project/vllm/issues/37431)), CUDA graph 캡처 중에는 `is_current_stream_capturing()` 체크로 건너뛰기 | `vllm/.../mamba/mamba_mixer2.py` |

### 빌드 시 적용 패치 (Dockerfile)

| 파일/패치 | 역할 |
|-----------|------|
| `patch_sm121_moe.py` | CuTe DSL `admissible_archs`에 `sm_120a`/`sm_121a` 추가 (18곳), TRT-LLM launcher `ICHECK_EQ(major, 10)` → `ICHECK_GE(major, 10)` |
| `patch_fp8_sm120.py` | SM120+에서 `FlashInferFP8ScaledMMLinearKernel` 차단 (segfault 방지) |
| `fix_quantization_utils_sm121.py` | `cvt.rn.satfinite.e2m1x2.f32` PTX 미지원 → 소프트웨어 E2M1 변환 |
| `scaled_fp4_quant` 패치 | vLLM `_C.abi3.so`의 SM121 sticky error 우회 → FlashInfer `nvfp4_quantize`로 대체 |
| PR #5823 패치 | `moe_gemm_tma_ws_launcher.inl` line 390: `== 100` → `>= 100` |
| CUTLASS 73c59c0 패치 | `sm120_blockscaled_mma_builder.inl`: `ReducedSmemCapacityBytes` grouped GEMM overhead 반영 |
| StageCount<2> 패치 | SM120 MoE grouped GEMM에서 2 pipeline stage 강제 |
| generate_kernels.py 패치 | SMEM 초과 타일 (128x128x256, 128x256x128, 256x128x128) 제거 |

---

## 7. 참조한 외부 이슈/PR/문서

### vLLM
| 이슈/PR | 제목 | 관련 |
|---------|------|------|
| [#21309](https://github.com/vllm-project/vllm/pull/21309) | Add CUTLASS NVFP4 SM120 kernels | SM120 FP4 GEMM 추가 (SMEM overflow 미해결) |
| [#32093](https://github.com/vllm-project/vllm/issues/32093) | FlashInfer SM120 지원 관련 | 유저가 참고용으로 공유 |
| [#33726](https://github.com/vllm-project/vllm/pull/33726) | Mamba2 + spec decode 지원 | `mamba_cache_mode=align` 도입 |
| [#34758](https://github.com/vllm-project/vllm/pull/34758) | vLLM main 빌드 관련 revert | eugr 소스 빌드 시 충돌 |
| [#35566](https://github.com/vllm-project/vllm/issues/35566) | SM120 CUTLASS FP4 MoE garbage output | `compute_120f` 필요 보고 |
| [#36094](https://github.com/vllm-project/vllm/issues/36094) | Qwen3.5 NVFP4 accuracy 저하 | SM100에서도 발생 — 모델 체크포인트 문제 가능성 |
| [#36821](https://github.com/vllm-project/vllm/issues/36821) | No sm_121 support on aarch64 — DGX Spark | SM121 aarch64 미지원 |
| [#36865](https://github.com/vllm-project/vllm/issues/36865) | SM120/RTX 5090 source build unsupported targets | 소스 빌드 아키텍처 문제 |
| [#37030](https://github.com/vllm-project/vllm/issues/37030) | MXFP4 on SM121 - Marlin kernel wrong tokens | SM121 Marlin 토큰 오류 |
| [#37141](https://github.com/vllm-project/vllm/issues/37141) | Upstream DGX Spark improvements from Avarok/dgx-vllm | DGX Spark 패치 upstream 요청 |
| [#33333](https://github.com/vllm-project/vllm/issues/33333) | FLASHINFER_CUTLASS not supported on SM120 | FlashInfer CUTLASS FP4 SM120 미지원 |
| [#33416](https://github.com/vllm-project/vllm/issues/33416) | NVFP4 MoE kernels fail on RTX Blackwell SM12.0 | MoE 커널 실패 보고 |
| [#36453](https://github.com/vllm-project/vllm/pull/36453) | SM120 패치 PR | `is_device_capability_family(120)` 추가 |
| [#37431](https://github.com/vllm-project/vllm/issues/37431) | Triton Mamba async bug on SM12x | `torch.cuda.synchronize()` 워크어라운드 필요 — 우리 mamba_mixer2_patch.py와 동일 문제 |
| [#21274](https://github.com/vllm-project/vllm/issues/21274) | nvfp4 support on sm120 | Closed (Not planned) — 초기 SM120 FP4 지원 요청 |

### FlashInfer
| 이슈/PR | 제목 | 관련 |
|---------|------|------|
| [#2252](https://github.com/flashinfer-ai/flashinfer/issues/2252) | FlashInfer SM120 지원 | 유저가 최초 참고용으로 공유 |
| [#2577](https://github.com/flashinfer-ai/flashinfer/issues/2577) | SM120 mm_fp4 all backends fail | CUTLASS all zeros, cuDNN 미지원. scale을 `.float()`로 변환 시 해결 제보 |
| [#2631](https://github.com/flashinfer-ai/flashinfer/pull/2631) | SM121 recognition 추가 | FlashInfer 0.6.6에 SM121 인식 포함, 하지만 FP4 GEMM 실행은 실패 |
| [#2670](https://github.com/flashinfer-ai/flashinfer/pull/2670) | SM120 FP4 shared memory fix (StageCount) | tinygemm2 SMEM 할당 축소 (FlashInfer 0.6.6) |
| [#2716](https://github.com/flashinfer-ai/flashinfer/pull/2716) | GDC barrier fix | PDL sync 수정 |
| [#2725](https://github.com/flashinfer-ai/flashinfer/pull/2725) | SM120 MoE backend selection 패치 PR | `is_device_capability_family(120)` 5개 파일 추가 |
| [#2786](https://github.com/flashinfer-ai/flashinfer/pull/2786) | Add K=64 tiles for SM120 MoE | K=64 타일 추가로 MoE 성능 2배 향상 |
| [#2835](https://github.com/flashinfer-ai/flashinfer/pull/2835) | SM120 SMEM arch detection (Python/CuTe DSL) | `get_blackwell_smem_arch()`로 SM120 vs SM100 감지 — 우리 수동 패치와 동일 문제 |

### CUTLASS
| 이슈/PR/커밋 | 제목 | 관련 |
|-------------|------|------|
| [#2185](https://github.com/NVIDIA/cutlass/pull/2185) | SM120 GEMM 수정 관련 | 유저가 참고용으로 공유 (v0.6.6에 포함 여부 확인) |
| [#2800](https://github.com/NVIDIA/cutlass/issues/2800) | BlockScaledMmaOp restricts FP4 to sm_100a only | SM120에서 FP4 MMA 사용 불가 원인 |
| [#2820](https://github.com/NVIDIA/cutlass/issues/2820) | SM120 Block-Scaled MMA Runtime Assertion Failure | StageCount assertion 실패 원인 분석 |
| [#3096](https://github.com/NVIDIA/cutlass/issues/3096) | SM120 FP4 GEMM performance | `compute_120f`로 39 tok/s, `compute_120a`로 14.6 tok/s. 10+ 패치 목록 문서화 |
| [73c59c0](https://github.com/NVIDIA/cutlass/commit/73c59c055c0fec87792470dbf33325158113db5e) | ReducedSmemCapacityBytes for grouped GEMM | SM120 grouped GEMM의 scheduler pipeline/tensor map overhead 반영 |

### TensorRT-LLM
| 이슈/PR | 제목 | 관련 |
|---------|------|------|
| [#5823](https://github.com/NVIDIA/TensorRT-LLM/pull/5823) | Fix MoE regression for SM120 | `== 100` → `>= 100` (C++ 레벨). merged but NOT propagated to FlashInfer |
| [#11368](https://github.com/NVIDIA/TensorRT-LLM/issues/11368) | FP4 CUTLASS GEMM fails on SM121 (shared memory overflow) | 타일 크기 128x128x256이 SM121 SMEM 초과 |
| [#11997](https://github.com/NVIDIA/TensorRT-LLM/pull/11997) | SM120/SM121 Python guard removal | `{100, 103}` → `{100, 103, 120, 121}` |
| [#12309](https://github.com/NVIDIA/TensorRT-LLM/pull/12309) | SM120 MoE backend support | `NotImplementedError` 가드 제거 |

### SGLang
| 이슈 | 제목 | 관련 |
|------|------|------|
| [#18954](https://github.com/sgl-project/sglang/issues/18954) | SM120 NVFP4 NaN output | FlashInfer CUTLASS/cuDNN 모두 NaN. FP8은 Triton GEMM 정상 |
| [#19637](https://github.com/sgl-project/sglang/issues/19637) | SM120 Performance Optimization Plan | 성능 최적화 로드맵 |
| [#20050](https://github.com/sgl-project/sglang/issues/20050) | TP8/TP4 gibberish, TP2 정상 | TP 설정 문제 (SM120 무관) |

### NVIDIA 공식 문서/포럼
| 링크 | 내용 |
|------|------|
| [Nemotron-3-Super Advanced Deployment Guide](https://docs.nvidia.com/nemotron/nightly/usage-cookbook/Nemotron-3-Super/AdvancedDeploymentGuide/README.html#vllm) | MTP `num_speculative_tokens > 1` 사용 예제. 유저가 "여기선 1보다 큰 값 쓰는데"로 참고 |
| [NVIDIA Forum: 50% improvement on Spark](https://forums.developer.nvidia.com/t/50-improvement-on-spark/363493) | DGX Spark 성능 50% 개선 사례 |
| [NVIDIA Forum: Run vLLM in Spark (page 4)](https://forums.developer.nvidia.com/t/run-vllm-in-spark/348862/61?page=4) | DGX Spark vLLM 실행 커뮤니티 스레드. `12.1f` 제안 |
| [NVIDIA Forum: Custom FP4 CUDA kernel 129 TFLOPS on DGX Spark](https://forums.developer.nvidia.com/t/custom-fp4-cuda-kernel-129-tflops-on-dgx-spark-with-pre-quantized-weight-cache/361600) | 커스텀 FP4 커널로 129 TFLOPS 달성 사례 |
| [NVIDIA Forum: "When are you going to fix tcgen05 FP4 for SM121"](https://forums.developer.nvidia.com/t/dearest-cutlass-team-when-the-hell-are-you-going-to-properly-fix-tcgen05-fp4-support-for-dgx-spark-gb10-sm121/359598) | SM121 FP4 지원 부재에 대한 커뮤니티 불만 |
| [NVIDIA Forum: SM121 software support lacking](https://forums.developer.nvidia.com/t/dgx-spark-sm121-software-support-is-severely-lacking-official-roadmap-needed/357663) | SM121 소프트웨어 지원 로드맵 요청 |
| [NVIDIA Forum: vLLM 0.17.0 MXFP4 patches for DGX Spark](https://forums.developer.nvidia.com/t/vllm-0-17-0-mxfp4-patches-for-dgx-spark-qwen3-5-35b-a3b-70-tok-s-gpt-oss-120b-80-tok-s-tp-2/362824) | Qwen3.5-35B 70 tok/s, GPT-OSS-120B 80 tok/s 달성 사례 |
| [NVIDIA Forum: SM121 CUTLASS kernel optimization](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960) | NVFP4 356 TFLOPS MoE grouped GEMM on DGX Spark. `sm_121a` → `BlockScaledMmaOp.admissible_archs` 추가 |
| [vLLM Benchmarking CLI Docs](https://docs.vllm.ai/en/latest/benchmarking/cli/#online-benchmark) | `vllm bench serve` 벤치마크 도구 문서 |
| [SemanticDiff: vLLM CMakeLists.txt SM120](https://app.semanticdiff.com/gh/vllm-project/vllm/commit/789562c28c143201a1d2ca35f7adcdf54ef832e5#CMakeLists.txt) | vLLM SM120 빌드 설정 변경 (유저 참고 공유) |

### 커뮤니티 프로젝트
| 프로젝트/링크 | 관련 |
|--------------|------|
| [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | DGX Spark용 prebuilt vLLM 이미지 |
| [eugr PR #98](https://github.com/eugr/spark-vllm-docker/pull/98) | Nemotron 지원 — `mamba-cache-mode=align`, Mamba SM121 recognition, CUTLASS 4.4.2 |
| [RobTand/spark-vllm-docker (flashinfer-pr-patching)](https://github.com/RobTand/spark-vllm-docker/tree/flashinfer-pr-patching) | E2M1 SM121 패치, FlashInfer K=64 SM120 패치 원본 |
| [Avarok-Cybersecurity/dgx-vllm](https://github.com/Avarok-Cybersecurity/dgx-vllm) | DGX Spark NVFP4 최적화 문서 (`NVFP4_BREAKTHROUGH_DGX_SPARK.md`) |
| [aliez-ren/vllm-qwen3.5-nvfp4-sm120](https://github.com/aliez-ren/vllm-qwen3.5-nvfp4-sm120) | Qwen3.5 NVFP4 SM120 최적화 |
| [JungkwanBan/SPARK_Qwen3.5-122B](https://github.com/JungkwanBan/SPARK_Qwen3.5-122B-A10B-NVFP4) | DGX Spark Qwen3.5 레시피 |
| [namake-taro/vllm-custom](https://github.com/namake-taro/vllm-custom) | SM120 커스텀 vLLM 빌드 |
| [Reddit: How I got Qwen3.5-397B running at speed](https://www.reddit.com/r/LocalLLaMA/comments/1rtrdsv/55_282_toks_how_i_got_qwen35397b_running_at_speed/) | Qwen3.5 NVFP4 최적화 사례 (Marlin + MTP) |
| [HuggingFace: super_v3_reasoning_parser.py](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) | Nemotron-3-Super 전용 reasoning parser |

### NVIDIA 기술 참고
| 링크 | 관련 |
|------|------|
| [CUDA PDL Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization) | GDC 플래그가 필요한 이유 — PDL sync |
| [PTX ISA: tcgen05-mma-scale-factor](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-b-layout-4x) | FP4 MMA scale factor 레이아웃 (E2M1 관련) |
| [CUTLASS Blackwell GeForce GEMM examples](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm) | SM120 전용 GEMM 예제 코드 |
| [TRT-LLM commit 52684d7](https://github.com/NVIDIA/TensorRT-LLM/commit/52684d79f7913973cd9e85f1a3de5c07ef01c039) | MoE SM120 수정 커밋 |
| [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) | Mamba2 causal-conv1d 네이티브 커널 (Triton 대체 시도) |
| [state-spaces/mamba v2.2.4](https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/selective_state_update.py) | Mamba SSM Triton 커널 원본 (SM121 async 버그 원인) |
| [Stacked DGX Sparks (build.nvidia.com)](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks) | DGX Spark 2대 연결 가이드 |
| [arxiv: Nemotron-3-Super paper (2512.12087)](https://arxiv.org/abs/2512.12087) | 모델 아키텍처 논문 |
| [arxiv: NemotronH architecture (2501.01005)](https://arxiv.org/abs/2501.01005) | NemotronH 하이브리드 아키텍처 상세 (Mamba+MoE+Attention) |
| [NVIDIA NeMo: Nemotron-3-Super vLLM cookbook](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Super/vllm_cookbook.ipynb) | 공식 vLLM 서빙 레시피 |
| [NVIDIA NeMo: Nemotron-3-Super README](https://github.com/NVIDIA-NeMo/Nemotron/blob/main/docs/nemotron/super3/README.md) | 모델 배포 가이드 |
| [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) | NVFP4 양자화 도구 |
| [CUDA Runtime API: cudaFuncSetAttribute](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html) | MaxDynamicSharedMemorySize 설정 참조 |
| [PTX ISA: cvt (data conversion)](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt) | E2M1 변환 PTX 명령어 참조 |
| [CUTLASS: SM120 BlockScaled Layout docs](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0y_blockscaled_layout.md) | FP4 BlockScaled 메모리 레이아웃 |
| [Avarok/dgx-vllm: NVFP4 Breakthrough](https://github.com/Avarok-Cybersecurity/dgx-vllm/blob/main/NVFP4_BREAKTHROUGH.md) | DGX Spark NVFP4 돌파 문서 |
| [Avarok/dgx-vllm: FlashInfer NVFP4 MoE fix](https://github.com/Avarok-Cybersecurity/dgx-vllm/blob/main/fix_flashinfer_nvfp4_moe_backend.py) | FlashInfer MoE 백엔드 수정 스크립트 |
| [Avarok/dgx-vllm: NVFP4 emulation fix](https://github.com/Avarok-Cybersecurity/dgx-vllm/blob/main/fix_nvfp4_emulation_backend.py) | NVFP4 에뮬레이션 백엔드 수정 |
| [kvcache-ai/custom_flashinfer](https://github.com/kvcache-ai/custom_flashinfer) | SM120 커스텀 FlashInfer 빌드 |
| [vLLM docs: torch_compile design](https://docs.vllm.ai/en/v0.10.0/design/v1/torch_compile.html) | torch.compile 통합 설계 문서 |
| [vLLM docs: compilation config](https://docs.vllm.ai/en/v0.10.1.1/api/vllm/config/compilation.html) | compilation_config 옵션 참조 |
| [vLLM docs: engine args](https://docs.vllm.ai/en/v0.10.2/configuration/engine_args.html) | 엔진 인자 전체 문서 |
| [PyTorch #122706](https://github.com/pytorch/pytorch/issues/122706) | functionalized tensor `data_ptr()` 호출 문제 (TVM DLPack 버그 관련) |

---

## 8. 최종 적용 구성

### docker-compose.yml 핵심 설정

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

## 9. 남은 한계점

### MoE Scatter Bandwidth — 하드 병목

모델의 40개 MoE 레이어에서 각 토큰마다 512개 전문가 중 22개를 읽어야 합니다.

- **순차 대역폭:** 241.5 GB/s
- **산란 읽기 유효 대역폭:** ~57-110 GB/s
- **L2 캐시:** 24 MB — 512개 전문가 가중치 캐싱 불가
- **결과:** 디코딩의 ~60%가 MoE 가중치 로딩에 소비

### 해결 불가능한 하드웨어 제약

1. **SM 48개** — max_autotune 불가, 병렬 제한
2. **SMEM 101KB/SM** — 큰 CUTLASS 타일 불가
3. **L2 24MB** — MoE expert 캐싱 불가
4. **nsys 미지원** — SM121 하드웨어 트레이서 한계

---

## 10. 전수 벤치마크 결과 (144개 조합, `vllm bench serve` 공식 CLI)

> **실행일:** 2026-03-22 ~ 2026-03-23
> **도구:** `vllm bench serve` (공식 CLI, [docs](https://docs.vllm.ai/en/latest/benchmarking/cli/#online-benchmark))
> **데이터셋:** ShareGPT (실제 대화 텍스트)
> **측정:** rate=1 (단일 사용자 latency) + rate=inf (최대 throughput)
> **조합:** GEMM(2) × Attention(2) × MTP(3) × CUDAGraph(3) × Fusion(2) × Eager(2) = 144
> **결과:** 84 OK, 60 SKIP (중복/불가능 조합 자동 스킵), 0 FAIL

### 범례

| 약어 | 의미 |
|------|------|
| **FC** | FlashInfer CUTLASS (네이티브 NVFP4 하드웨어 FP4) |
| **Marlin** | Marlin W4A16 (FP4→BF16 디퀀트 후 BF16 GEMM) |
| **FI** | FlashInfer Attention (JIT sm_121a) |
| **Triton** | TRITON_ATTN |
| **f&p** | `full_and_piecewise` CUDA graph mode |
| **piece** | `piecewise` CUDA graph mode |
| **none** | CUDA graph 없음 (torch.compile만) |
| **Eager** | `--enforce-eager` (torch.compile + CUDA graph 전부 비활성화) |
| **ON/OFF** | `fuse_norm_quant` + `fuse_act_quant` 커널 퓨전 |

### 전체 결과 (Single tok/s 내림차순)

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

### 핵심 발견

**1. 최적 조합: `Marlin + TRITON_ATTN + MTP=2 + full_and_piecewise + fusion ON` = 46.3 tok/s**

- 차선: `Marlin + FlashInfer + MTP=2 + f&p + fusion ON` = 45.3 tok/s (거의 동일)
- FlashInfer CUTLASS 최고: `FC + TRITON_ATTN + MTP=2 + f&p + fusion ON` = 44.7 tok/s (3위)

**2. Marlin > FlashInfer CUTLASS (동일 조건 비교 시 평균 +10~15%)**

- SM121의 101KB SMEM에서 CUTLASS FP4 BlockScaled는 128×128×64B 타일 1종 + 2 stages만 가능
- TMA Warp Specialized 80개 tactic 전부 SMEM overflow로 실패
- Marlin W4A16은 BF16 텐서코어 사용, SMEM 제약 적음

**3. MTP 효과: MTP=0 → MTP=2로 +35~45% 향상**

| MTP | 평균 Single tok/s (상위 10개) | 대표 TPOT |
|-----|---------------------------|-----------|
| 0 | 31.2 | 137ms |
| 1 | 39.7 | 118ms |
| 2 | 41.6 | 116ms |

**4. CUDA graph 모드 효과**

| Mode | 평균 tok/s (MTP=2 조합) | 비고 |
|------|----------------------|------|
| full_and_piecewise | 40.3 | 최적 (Mamba + MTP 호환) |
| none (torch.compile) | 38.1 | graph overhead 없지만 launch 비용 |
| piecewise | 37.4 | f&p 대비 약간 느림 |
| enforce-eager | 38.5 | compile 없이도 선전 |

**5. Fusion 효과: ON이 OFF보다 평균 +5~10%**

**6. Attention 백엔드: TRITON_ATTN ≈ FLASHINFER (차이 미미)**

- 상위 10개 중 두 백엔드가 거의 교차 출현
- FlashInfer가 TPOT에서 약간 유리 (105ms vs 117ms), TRITON이 throughput에서 약간 유리

---

## 11. 요약

| 메트릭 | 값 |
|--------|-----|
| **최적 조합** | **Marlin + MTP=2 + f&p + fusion ON** |
| **단일 요청 (ShareGPT)** | **46.3 tok/s** |
| **최대 throughput (ShareGPT)** | **63.3 tok/s** |
| **TPOT** | **116.9 ms** |
| **전수 벤치마크** | 144개 조합 (84 OK, 60 SKIP, 0 FAIL) |
| **Marlin vs FC 비교** | Marlin이 평균 +10~15% 빠름 |
| **MTP 효과** | MTP=0→2로 +35~45% |
| **적용된 패치** | 9+ 파일, 6+ SM121 버그 수정 |
| **참조한 외부 이슈/PR** | 30+ 건 |
| **하드 병목** | MoE scatter bandwidth (~110 GB/s effective) |
