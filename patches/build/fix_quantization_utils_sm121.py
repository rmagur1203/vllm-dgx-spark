#!/usr/bin/env python3
"""Add software E2M1 conversion fallback to quantization_utils.cuh for SM121.

SM121 (GB10) lacks cvt.rn.satfinite.e2m1x2.f32 PTX. We add software E2M1
helpers and insert early-return guards in the 3 fp32_vec_to_e2m1 overloads.

IMPORTANT: We do NOT modify the >= 1000 guards on higher-level wrapper functions
(cvt_warp_fp16_to_fp4, cvt_warp_fp8_to_fp4, etc.) — those contain generic float
math (scale factors, type conversion) that works fine on SM121. They call
fp32_vec_to_e2m1 which has our software fallback.

The previous approach of globally adding (!= 1210) to ALL (__CUDA_ARCH__ >= 1000)
guards was WRONG — it caused wrapper functions to skip entirely on SM121, leaving
scale factors unwritten (garbage) → 0 * NaN = NaN in the GEMM.
"""

import sys

path = "csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh"

with open(path) as f:
    content = f.read()

if "_sw_e2m1_single" in content:
    print("Software E2M1 fallback already present")
    sys.exit(0)

# 1. Add software E2M1 helper functions (guarded by __CUDA_ARCH__ == 1210)
SW_E2M1_HELPER = r"""
// Software E2M1 conversion for SM121 (GB10) - no cvt.rn.satfinite.e2m1x2.f32
// Reference: https://github.com/Avarok-Cybersecurity/dgx-vllm
__device__ inline uint8_t _sw_e2m1_single(float v) {
    uint8_t sign = (__float_as_uint(v) >> 31) & 1;
    float av = fabsf(v);
    uint8_t e2m1;
    if      (av < 0.25f) e2m1 = 0;
    else if (av < 0.75f) e2m1 = 1;
    else if (av < 1.25f) e2m1 = 2;
    else if (av < 1.75f) e2m1 = 3;
    else if (av < 2.5f)  e2m1 = 4;
    else if (av < 3.5f)  e2m1 = 5;
    else if (av < 5.0f)  e2m1 = 6;
    else                 e2m1 = 7;
    return (sign << 3) | e2m1;
}
__device__ inline uint8_t _sw_e2m1x2(float lo, float hi) {
    return (_sw_e2m1_single(lo) & 0xF) | ((_sw_e2m1_single(hi) & 0xF) << 4);
}
__device__ inline uint32_t _sw_e2m1x8(float* a) {
    return (uint32_t)_sw_e2m1x2(a[0],a[1]) | ((uint32_t)_sw_e2m1x2(a[2],a[3])<<8) |
           ((uint32_t)_sw_e2m1x2(a[4],a[5])<<16) | ((uint32_t)_sw_e2m1x2(a[6],a[7])<<24);
}
__device__ inline uint32_t _sw_e2m1x8_f2(float2* a) {
    float f[8]={a[0].x,a[0].y,a[1].x,a[1].y,a[2].x,a[2].y,a[3].x,a[3].y};
    return _sw_e2m1x8(f);
}
__device__ inline uint64_t _sw_e2m1x16_f2(float2* a) {
    float f0[8]={a[0].x,a[0].y,a[1].x,a[1].y,a[2].x,a[2].y,a[3].x,a[3].y};
    float f1[8]={a[4].x,a[4].y,a[5].x,a[5].y,a[6].x,a[6].y,a[7].x,a[7].y};
    return (uint64_t)_sw_e2m1x8(f0) | ((uint64_t)_sw_e2m1x8(f1)<<32);
}
"""

# Insert helper functions before the first fp32_vec_to_e2m1 definition
marker = "// Convert 8 float32 values into 8 e2m1 values"
if marker not in content:
    print(f"WARNING: marker not found in {path}")
    sys.exit(1)

content = content.replace(marker, SW_E2M1_HELPER + "\n" + marker, 1)
print("Inserted software E2M1 helper functions")

# 2. For each fp32_vec_to_e2m1 overload, insert SM121 early-return INSIDE the
#    #if >= 1000 block, right after the opening brace + #if guard.
#    This way SM121 enters the >= 1000 path (so wrapper functions work normally)
#    but the actual PTX E2M1 conversion is replaced with software.
#
#    Pattern in the source:
#      inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
#      #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#        uint32_t val;
#        asm volatile(...)  // <-- the PTX that SM121 can't do
#
#    We insert after the #if line:
#      #if __CUDA_ARCH__ == 1210
#        return _sw_e2m1x8(array);  // software fallback
#      #endif

replacements = [
    # (function signature fragment, software call)
    ("fp32_vec_to_e2m1(float (&array)[8]) {", "  return _sw_e2m1x8(array);"),
    ("fp32_vec_to_e2m1(float2 (&array)[4]) {", "  return _sw_e2m1x8_f2(array);"),
    ("fp32_vec_to_e2m1(float2 (&array)[8]) {", "  return _sw_e2m1x16_f2(array);"),
]

patched = 0
for sig, sw_return in replacements:
    # Find the function, then find the #if >= 1000 guard after it
    idx = content.find(sig)
    if idx < 0:
        print(f"  WARNING: '{sig}' not found")
        continue

    # Find the #if >= 1000 line after the function signature
    guard = "#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)"
    guard_idx = content.find(guard, idx)
    if guard_idx < 0 or guard_idx > idx + 200:
        print(f"  WARNING: #if guard not found near '{sig}'")
        continue

    # Insert SM121 early return right after the #if guard line
    insert_pos = content.find("\n", guard_idx) + 1
    sm121_block = f"#if __CUDA_ARCH__ == 1210\n{sw_return}\n#else\n"

    # Also need to close the #else before the existing #else/#endif
    # Find the matching #else for this #if block
    # The pattern is: #if >= 1000 ... code ... #else ... return 0 ... #endif
    existing_else = content.find("#else", insert_pos)
    if existing_else > 0 and existing_else < insert_pos + 2000:
        # Insert #endif to close our #if/#else before the existing #else
        content = (content[:insert_pos] + sm121_block +
                   content[insert_pos:existing_else] +
                   "#endif  // SM121\n" +
                   content[existing_else:])
        patched += 1
        print(f"  Patched: {sig}")
    else:
        print(f"  WARNING: could not find #else for '{sig}'")

print(f"\nInserted SM121 early-return in {patched}/3 fp32_vec_to_e2m1 functions")

with open(path, "w") as f:
    f.write(content)

if patched < 3:
    print("WARNING: not all functions were patched!")
    sys.exit(1)

print(f"Successfully patched {path}")
