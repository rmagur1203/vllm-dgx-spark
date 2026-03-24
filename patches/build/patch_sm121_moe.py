"""
Patch FlashInfer 0.6.6 for SM121 (DGX Spark) native NVFP4 MoE support.

Applies the following fixes from CUTLASS #3096 / TRT-LLM PR #5823:
1. CuTe DSL admissible_archs: Add sm_120a, sm_120f, sm_121a to all arch lists
2. FP8 ScaledMM: Block sm120+ (FlashInfer & CUTLASS FP8 GEMM broken on SM121)
3. TRT-LLM launcher: EQUAL 100 -> GREATER_EQUAL 100 equivalent
"""
import os
import sys
import py_compile
import glob

FI_ROOT = '/usr/local/lib/python3.12/dist-packages/flashinfer'
VLLM_ROOT = '/usr/local/lib/python3.12/dist-packages/vllm'
CUTE_DSL = os.path.join(FI_ROOT, 'data/cutlass/python/CuTeDSL/cutlass/cute')

patched = []
errors = []

def patch_file(path, old, new, description, all_occurrences=False):
    """Patch a file, replacing old with new."""
    try:
        with open(path) as f:
            content = f.read()
        if new in content:
            print(f'  [SKIP] {description} - already patched')
            return True
        if old not in content:
            print(f'  [WARN] {description} - pattern not found in {path}')
            return False
        if all_occurrences:
            content = content.replace(old, new)
        else:
            content = content.replace(old, new, 1)
        with open(path, 'w') as f:
            f.write(content)
        py_compile.compile(path, doraise=True)
        patched.append(description)
        print(f'  [OK]   {description}')
        return True
    except Exception as e:
        errors.append(f'{description}: {e}')
        print(f'  [ERR]  {description}: {e}')
        return False


# ============================================================
# 1. CuTe DSL: Add SM120/SM121 to admissible_archs
# ============================================================
print('\n=== Patching CuTe DSL admissible_archs ===')

# SM120/SM121 are Blackwell-family and support the same tcgen05 instructions as SM100
sm120_archs = '"sm_120a", "sm_120f", "sm_121a"'

cute_files = [
    ('nvgpu/tcgen05/mma.py', '"sm_100a",\n        "sm_100f",', f'"sm_100a",\n        "sm_100f",\n        {sm120_archs},'),
    ('nvgpu/tcgen05/mma.py', '"sm_100a",\n    ]', f'"sm_100a",\n        {sm120_archs},\n    ]'),
    ('nvgpu/tcgen05/copy.py', '"sm_100a",\n        "sm_100f",', f'"sm_100a",\n        "sm_100f",\n        {sm120_archs},'),
    ('nvgpu/cpasync/copy.py', '"sm_100a",\n        "sm_100f",', f'"sm_100a",\n        "sm_100f",\n        {sm120_archs},'),
]

for relpath, old, new in cute_files:
    filepath = os.path.join(CUTE_DSL, relpath)
    if os.path.exists(filepath):
        patch_file(filepath, old, new, f'CuTe DSL {relpath}', all_occurrences=True)
    else:
        print(f'  [WARN] File not found: {filepath}')

# Also patch arch/mbar.py and arch/elect.py if they have admissible_archs
for archfile in ['arch/mbar.py', 'arch/elect.py']:
    filepath = os.path.join(CUTE_DSL, archfile)
    if os.path.exists(filepath):
        with open(filepath) as f:
            content = f.read()
        if 'admissible_archs' in content and 'sm_121a' not in content:
            patch_file(filepath, '"sm_100a",\n        "sm_100f",',
                      f'"sm_100a",\n        "sm_100f",\n        {sm120_archs},',
                      f'CuTe DSL {archfile}', all_occurrences=True)

# ============================================================
# 2. FlashInfer FP8 ScaledMM: Block SM120+
# ============================================================
print('\n=== Patching FP8 ScaledMM for SM120+ ===')

fi_fp8 = os.path.join(VLLM_ROOT, 'model_executor/kernels/linear/scaled_mm/flashinfer.py')
if os.path.exists(fi_fp8):
    patch_file(fi_fp8,
        'compute_capability < 100',
        'compute_capability < 100 or compute_capability >= 120',
        'FlashInfer FP8 ScaledMM sm120 guard')

cu_fp8 = os.path.join(VLLM_ROOT, 'model_executor/kernels/linear/scaled_mm/cutlass.py')
if os.path.exists(cu_fp8):
    with open(cu_fp8) as f:
        content = f.read()
    old_block = '        if not current_platform.is_cuda():\n            return False, "requires CUDA."\n        return True, None'
    new_block = '        if not current_platform.is_cuda():\n            return False, "requires CUDA."\n        if compute_capability is not None and compute_capability >= 120:\n            return False, "CUTLASS FP8 lacks sm120 kernel images."\n        return True, None'
    if new_block not in content:
        idx1 = content.find(old_block)
        if idx1 >= 0:
            idx2 = content.find(old_block, idx1 + 1)
            if idx2 >= 0:
                content = content[:idx2] + content[idx2:].replace(old_block, new_block, 1)
                with open(cu_fp8, 'w') as f:
                    f.write(content)
                py_compile.compile(cu_fp8, doraise=True)
                patched.append('CUTLASS FP8 ScaledMM sm120 guard')
                print('  [OK]   CUTLASS FP8 ScaledMM sm120 guard')
            else:
                print('  [WARN] CUTLASS FP8: second occurrence not found')
        else:
            print('  [WARN] CUTLASS FP8: pattern not found')
    else:
        print('  [SKIP] CUTLASS FP8 ScaledMM - already patched')

# ============================================================
# 3. TRT-LLM fused MoE launcher: SM check relaxation
# ============================================================
print('\n=== Patching TRT-LLM fused MoE launcher ===')

# Check for C++ launcher files that might have SM100 checks
tllm_launcher_files = glob.glob(os.path.join(FI_ROOT, 'data/include/**/*moe*'), recursive=True) + \
                      glob.glob(os.path.join(FI_ROOT, 'data/include/**/*launcher*'), recursive=True)

for lf in tllm_launcher_files:
    if lf.endswith(('.h', '.cu', '.cuh')):
        with open(lf) as f:
            content = f.read()
        # TRT-LLM PR #5823: EQUAL 100 -> GREATER_EQUAL 100
        if '== 100' in content or '== SM100' in content:
            print(f'  [INFO] Found SM100 check in {lf}')
            # Replace == 100 with >= 100 in architecture checks
            new_content = content.replace('== 100', '>= 100')
            if new_content != content:
                with open(lf, 'w') as f:
                    f.write(new_content)
                patched.append(f'TRT-LLM launcher {os.path.basename(lf)}')
                print(f'  [OK]   Patched {os.path.basename(lf)}')

# Also check for arch == sm_100 type checks in Python
tllm_py = os.path.join(FI_ROOT, 'jit/tllm_utils.py')
if os.path.exists(tllm_py):
    with open(tllm_py) as f:
        content = f.read()
    if '== 100' in content or "sm_100" in content:
        print(f'  [INFO] Found SM100 check in tllm_utils.py')

# ============================================================
# Summary
# ============================================================
print(f'\n=== Summary ===')
print(f'Patched: {len(patched)} files')
for p in patched:
    print(f'  + {p}')
if errors:
    print(f'Errors: {len(errors)}')
    for e in errors:
        print(f'  ! {e}')
