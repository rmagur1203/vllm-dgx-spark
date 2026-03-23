#!/usr/bin/env python3
"""
Exhaustive benchmark: 144 config combinations for Nemotron-3-Super-120B on DGX Spark.
2×2×3×3×2×2 = 144 (GEMM × Attention × MTP × CUDAGraph × Fusion × Eager)

Uses: vllm bench serve (official CLI) with ShareGPT dataset (real text).
Ref: https://docs.vllm.ai/en/latest/benchmarking/cli/#online-benchmark
"""
import itertools, json, os, subprocess, sys, time

API_KEY = "${VLLM_API_KEY}"
CONTAINER = "vllm-nemotron-3-super-120b-a12b-nvfp4"
TOKENIZER = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
RESULTS = "/home/user/vllm/bench_results.jsonl"
COMPOSE = "/home/user/vllm/docker-compose.yml"

# ============================================================
# 6 axes → 144 combinations
# ============================================================
AXES = {
    'gemm':   ['flashinfer-cutlass', 'marlin'],
    'attn':   ['FLASHINFER', 'TRITON_ATTN'],
    'mtp':    [0, 1, 2],
    'cg':     ['none', 'piecewise', 'full_and_piecewise'],
    'fusion': [True, False],
    'eager':  [True, False],
}

# ============================================================
# Docker Compose generation
# ============================================================
VOLUMES = """      - ./hf-cache:/root/.cache
      - ./prebuild_flashinfer_jit.py:/workspace/prebuild_flashinfer_jit.py:ro
      - ./tvm_build_patch.py:/usr/local/lib/python3.12/dist-packages/tvm_ffi/utils/_build_optional_torch_c_dlpack.py:ro
      - ./cutlass_heuristic_patch.cpp:/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/cutlass_heuristic.cpp:ro
      - ./vllm_flashinfer_utils_patch.py:/usr/local/lib/python3.12/dist-packages/vllm/utils/flashinfer.py:ro
      - ./flashinfer_backend_patch.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/flashinfer.py:ro
      - ./mamba_mixer2_patch.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/mamba_mixer2.py:ro
      - ./super_v3_reasoning_parser.py:/workspace/super_v3_reasoning_parser.py:ro
      - ./sonnet.txt:/tmp/sonnet.txt:ro
      - ./sharegpt.json:/tmp/sharegpt.json:ro"""


def write_compose(gemm, attn, mtp, cg, fusion, eager):
    max_batch = 8400 if mtp == 2 else 8352

    env_lines = []
    if gemm == 'marlin':
        env_lines.append('      VLLM_NVFP4_GEMM_BACKEND: "marlin"')
        env_lines.append('      VLLM_TEST_FORCE_FP8_MARLIN: "1"')
        env_lines.append('      VLLM_USE_FLASHINFER_MOE_FP4: "0"')
    else:
        env_lines.append('      VLLM_NVFP4_GEMM_BACKEND: "flashinfer-cutlass"')
        env_lines.append('      VLLM_USE_FLASHINFER_MOE_FP4: "1"')
    env_block = '\n'.join(env_lines)

    cmd = []
    cmd.append(f'      - --attention-backend={attn}')
    cmd.append(f'      - --max-num-batched-tokens={max_batch}')

    if eager:
        cmd.append('      - --enforce-eager')
    else:
        cc_parts = [f'"cudagraph_mode":"{cg}"']
        if fusion:
            cc_parts.append('"pass_config":{"fuse_norm_quant":true,"fuse_act_quant":true}')
        cmd.append('      - --compilation-config={' + ','.join(cc_parts) + '}')

    if mtp > 0:
        cmd.append(f'      - --speculative-config={{"method":"mtp","num_speculative_tokens":{mtp}}}')

    cmd_block = '\n'.join(cmd)

    compose = f'''services:
  drop-caches:
    image: alpine
    privileged: true
    pid: host
    entrypoint: ["sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches && echo 'Page cache dropped'"]

  nemotron-3-super:
    build:
      context: .
      dockerfile: vllm.Dockerfile
    image: vllm:latest
    container_name: {CONTAINER}
    depends_on:
      drop-caches:
        condition: service_completed_successfully
    restart: "no"
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
    ipc: host
    shm_size: "16g"
    ulimits:
      memlock: -1
      stack: 67108864
    environment:
      TORCHINDUCTOR_COMPILE_THREADS: "1"
      MAX_JOBS: "2"
      FLASHINFER_CUDA_ARCH_LIST: "12.1a"
      FLASHINFER_DISABLE_VERSION_CHECK: "1"
      FLASHINFER_EXTRA_CUDAFLAGS: "-DCUTLASS_ENABLE_GDC_FOR_SM100=1"
      VLLM_KV_CACHE_LAYOUT: "HND"
{env_block}
      HF_HOME: "/root/.cache/huggingface"
    volumes:
{VOLUMES}
    entrypoint: ["bash", "-c", "python3 /workspace/prebuild_flashinfer_jit.py && exec python3 -m vllm.entrypoints.openai.api_server \\"$$@\\"", "--"]
    command:
      - --model=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
      - --served-model-name=vllm/nemotron-3-super-120b
      - --trust-remote-code
      - --enable-sleep-mode
      - --tool-call-parser=qwen3_coder
      - --reasoning-parser-plugin=/workspace/super_v3_reasoning_parser.py
      - --reasoning-parser=super_v3
      - --enable-auto-tool-choice
      - --host=0.0.0.0
      - --port=8000
      - --tensor-parallel-size=1
      - --enable-chunked-prefill
      - --max-model-len=131072
      - --enable-prefix-caching
      - --mamba-cache-mode=align
      - --load-format=fastsafetensors
      - --gpu-memory-utilization=0.8
      - --attention-config={{"disable_flashinfer_q_quantization":true}}
{cmd_block}
      - --api-key={API_KEY}
'''
    with open(COMPOSE, 'w') as f:
        f.write(compose)


# ============================================================
# Server lifecycle
# ============================================================
def restart_server():
    subprocess.run(['docker', 'compose', 'stop', 'nemotron-3-super'],
                   capture_output=True, timeout=120)
    subprocess.run(['docker', 'compose', 'rm', '-f', 'nemotron-3-super'],
                   capture_output=True, timeout=30)
    time.sleep(5)
    r = subprocess.run(['docker', 'compose', 'up', '-d', 'nemotron-3-super'],
                       capture_output=True, text=True, timeout=60)
    return r.returncode == 0, r.stderr[-500:] if r.returncode else ''


def wait_for_server(timeout=900):
    import urllib.request
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            req = urllib.request.Request('http://localhost:8000/health',
                headers={'Authorization': f'Bearer {API_KEY}'})
            if urllib.request.urlopen(req, timeout=5).status == 200:
                return True
        except:
            pass
        time.sleep(15)
    return False


def get_error_log():
    r = subprocess.run(['docker', 'logs', '--tail', '30', CONTAINER],
                       capture_output=True, text=True, timeout=10)
    return (r.stdout + r.stderr)[-500:]


def warmup_server():
    import urllib.request
    for prompt in ["Hello", "Count from 1 to 5"]:
        try:
            data = json.dumps({'model': 'vllm/nemotron-3-super-120b',
                'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 20}).encode()
            req = urllib.request.Request('http://localhost:8000/v1/chat/completions', data=data,
                headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {API_KEY}'})
            urllib.request.urlopen(req, timeout=120)
        except:
            pass


# ============================================================
# Benchmark runner (ShareGPT dataset, real text)
# ============================================================
def run_bench(dataset='sharegpt', rate='1', num_prompts=10, warmups=2):
    cmd = ['docker', 'exec', '-e', f'OPENAI_API_KEY={API_KEY}', CONTAINER,
           'vllm', 'bench', 'serve',
           '--backend', 'openai', '--base-url', 'http://localhost:8000',
           '--model', 'vllm/nemotron-3-super-120b',
           '--tokenizer', TOKENIZER,
           '--dataset-name', dataset,
           '--num-prompts', str(num_prompts),
           '--request-rate', rate,
           '--num-warmups', str(warmups)]

    if dataset == 'sharegpt':
        cmd += ['--dataset-path', '/tmp/sharegpt.json']
    elif dataset == 'sonnet':
        cmd += ['--dataset-path', '/tmp/sonnet.txt']
    elif dataset == 'random':
        cmd += ['--input-len', '128', '--output-len', '256']

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    output = r.stdout + r.stderr
    m = {}
    for line in output.split('\n'):
        line = line.strip()
        for lbl, key in [
            ('Output token throughput', 'output_tps'),
            ('Total token throughput', 'total_tps'),
            ('Mean TPOT', 'mean_tpot_ms'),
            ('Mean ITL', 'mean_itl_ms'),
            ('P99 ITL', 'p99_itl_ms'),
            ('Mean TTFT', 'mean_ttft_ms'),
            ('Peak output', 'peak_output_tps'),
            ('Acceptance rate', 'mtp_accept_pct'),
            ('Acceptance length', 'mtp_accept_len'),
        ]:
            if lbl in line and (lbl != 'Peak output' or 'Peak output' in line):
                try:
                    m[key] = float(line.split()[-1])
                except:
                    pass
    return m


# ============================================================
# Label and resume
# ============================================================
def make_label(gemm, attn, mtp, cg, fusion, eager):
    parts = [f'gemm={gemm}', f'attn={attn}', f'mtp={mtp}']
    if eager:
        parts.append('eager=ON')
    else:
        parts.append(f'cg={cg}')
        parts.append(f'fuse={"ON" if fusion else "OFF"}')
    return ' '.join(parts)


def already_done(label):
    if not os.path.exists(RESULTS):
        return False
    with open(RESULTS) as f:
        for line in f:
            try:
                if json.loads(line).get('label') == label:
                    return True
            except:
                pass
    return False


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    keys = list(AXES.keys())
    combos = list(itertools.product(*[AXES[k] for k in keys]))
    print(f"Total combinations: {len(combos)}")
    print(f"Dataset: ShareGPT (real text conversations)")
    print(f"Benchmark: vllm bench serve (official CLI)")
    print()

    done = skipped = failed = 0

    for i, vals in enumerate(combos, 1):
        cfg = dict(zip(keys, vals))
        label = make_label(**cfg)

        if already_done(label):
            print(f"[{i}/{len(combos)}] SKIP: {label}")
            skipped += 1
            continue

        print(f"\n{'='*70}")
        print(f"[{i}/{len(combos)}] {label}")
        print(f"{'='*70}")

        write_compose(**cfg)

        print("  Restarting...", flush=True)
        ok, err = restart_server()
        if not ok:
            print(f"  COMPOSE ERROR: {err[:200]}")
            with open(RESULTS, 'a') as f:
                f.write(json.dumps({'label': label, 'config': cfg, 'status': 'COMPOSE_ERROR', 'error': err[:500]}) + '\n')
            failed += 1
            continue

        print("  Waiting for server...", end=' ', flush=True)
        if not wait_for_server():
            err = get_error_log()
            print("TIMEOUT")
            with open(RESULTS, 'a') as f:
                f.write(json.dumps({'label': label, 'config': cfg, 'status': 'TIMEOUT', 'error': err}) + '\n')
            failed += 1
            continue
        print("READY")

        warmup_server()

        try:
            # ShareGPT: rate=1 (single user)
            print("  ShareGPT rate=1...", end=' ', flush=True)
            m1 = run_bench(dataset='sharegpt', rate='1', num_prompts=10, warmups=2)
            print(f"output={m1.get('output_tps', '?')} tok/s, tpot={m1.get('mean_tpot_ms', '?')}ms")

            # ShareGPT: rate=inf (max throughput)
            print("  ShareGPT rate=inf...", end=' ', flush=True)
            m2 = run_bench(dataset='sharegpt', rate='inf', num_prompts=20, warmups=0)
            print(f"output={m2.get('output_tps', '?')} tok/s")

            result = {'label': label, 'config': cfg, 'status': 'OK',
                      'sharegpt_single': m1, 'sharegpt_concurrent': m2}
            done += 1
        except Exception as e:
            print(f"BENCH ERROR: {e}")
            result = {'label': label, 'config': cfg, 'status': 'BENCH_ERROR', 'error': str(e)}
            failed += 1

        with open(RESULTS, 'a') as f:
            f.write(json.dumps(result) + '\n')

        # Progress report
        total_done = done + failed + skipped
        eta_min = (len(combos) - total_done) * 10
        print(f"  Progress: {total_done}/{len(combos)} ({done} OK, {failed} FAIL, {skipped} SKIP) ETA ~{eta_min}min")

    # Restore optimal
    print(f"\n{'='*70}")
    print(f"COMPLETE: {done} OK, {failed} FAIL, {skipped} SKIP out of {len(combos)}")
    print(f"{'='*70}")
    print("Restoring optimal config...")
    write_compose(gemm='flashinfer-cutlass', attn='FLASHINFER', mtp=2,
                  cg='full_and_piecewise', fusion=True, eager=False)
    restart_server()
    wait_for_server()
    print("Done.")
