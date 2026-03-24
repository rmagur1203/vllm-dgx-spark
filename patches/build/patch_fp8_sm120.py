"""Patch FlashInfer and CUTLASS FP8 ScaledMM to reject SM120+ (DGX Spark)."""
import py_compile

# 1. FlashInfer FP8 ScaledMM - add compute_capability >= 120 guard
fi_path = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py'
with open(fi_path) as f:
    content = f.read()
old = 'compute_capability < 100'
new = 'compute_capability < 100 or compute_capability >= 120'
if old in content and new not in content:
    content = content.replace(old, new)
    content = content.replace(
        'requires compute capability 100 and above.',
        'requires compute capability 100-119.')
    with open(fi_path, 'w') as f:
        f.write(content)
    py_compile.compile(fi_path, doraise=True)
    print('FlashInfer FP8 ScaledMM patched for SM120+')
else:
    print('FlashInfer FP8 ScaledMM already patched or not found')

# 2. CUTLASS FP8 ScaledMM - add compute_capability >= 120 guard (2nd is_supported)
cu_path = '/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py'
with open(cu_path) as f:
    content = f.read()
old_block = '        if not current_platform.is_cuda():\n            return False, "requires CUDA."\n        return True, None'
new_block = '        if not current_platform.is_cuda():\n            return False, "requires CUDA."\n        if compute_capability is not None and compute_capability >= 120:\n            return False, "CUTLASS FP8 lacks sm120 kernel images."\n        return True, None'
if new_block not in content:
    # Replace only the second occurrence (FP8 class)
    idx1 = content.find(old_block)
    if idx1 >= 0:
        idx2 = content.find(old_block, idx1 + 1)
        if idx2 >= 0:
            content = content[:idx2] + content[idx2:].replace(old_block, new_block, 1)
            with open(cu_path, 'w') as f:
                f.write(content)
            py_compile.compile(cu_path, doraise=True)
            print('CUTLASS FP8 ScaledMM patched for SM120+')
        else:
            print('CUTLASS FP8: second occurrence not found, skipping')
    else:
        print('CUTLASS FP8: pattern not found, skipping')
else:
    print('CUTLASS FP8 ScaledMM already patched')
