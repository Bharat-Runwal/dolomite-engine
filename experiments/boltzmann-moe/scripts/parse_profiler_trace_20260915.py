#!/usr/bin/env python3
"""Parse the scale32B_boltz_hop_PROFILE torch.profiler trace and attribute GPU time.

The trace has no module annotations (profiler built without with_modules/with_stack),
so we attribute by (a) kernel-name family and (b) GEMM input shapes, which are enough
to separate the dense-MoE block (I_e=1280 / total 40960 experts) from the GPT-prefix
FFN (swiglu 4096), attention, the lm_head (vocab 100352), and FSDP comms.

Pure stdlib; run directly on a compute node (no GPU, no numpy needed).
"""
import json, sys, re
from collections import defaultdict

path = sys.argv[1]
blob = json.load(open(path))
ev = blob["traceEvents"]
print(f"total trace events: {len(ev):,}")

# --- categories present ---
cats = defaultdict(int)
for e in ev:
    cats[e.get("cat", "?")] += 1
print("categories:", dict(sorted(cats.items(), key=lambda x: -x[1])))

# --- GPU kernels: cat == 'kernel' (device side). Sum dur (microseconds). ---
kern_time = defaultdict(float)   # name -> us
kern_count = defaultdict(int)
total_kernel_us = 0.0
for e in ev:
    if e.get("cat") == "kernel":
        d = e.get("dur", 0.0)
        total_kernel_us += d
        kern_time[e["name"]] += d
        kern_count[e["name"]] += 1
print(f"\ntotal GPU kernel time (active window): {total_kernel_us/1000:.2f} ms")
print(f"distinct kernels: {len(kern_time)}")

# --- functional buckets by kernel-name family ---
def bucket(name):
    n = name.lower()
    if any(k in n for k in ("nccl", "allgather", "all_gather", "reduce_scatter",
                            "reducescatter", "allreduce", "all_reduce")):
        return "FSDP_comms"
    # attention BEFORE gemm: cudnn flash kernels carry 'sm90' in their name and would
    # otherwise be miscounted as GEMM.
    if any(k in n for k in ("flash", "attention", "mha", "fmha", "scaled_dot",
                            "sdpa", "_softmax", "softmax_warp")):
        return "attention"
    if any(k in n for k in ("nvjet", "gemm", "cutlass", "cublas", "ampere", "sm80", "sm90",
                            "wgmma", "s16816", "h16816", "gett")):
        return "GEMM"
    if "triton" in n:
        return "triton_fused"
    if any(k in n for k in ("elementwise", "vectorized_elementwise", "pointwise")):
        return "elementwise"
    if any(k in n for k in ("reduce", "norm", "layer_norm", "rms")):
        return "reduce/norm"
    if any(k in n for k in ("memcpy", "memset", "copy")):
        return "copy/memset"
    return "other"

buck = defaultdict(float)
for name, t in kern_time.items():
    buck[bucket(name)] += t
print("\n=== GPU time by kernel family ===")
for b, t in sorted(buck.items(), key=lambda x: -x[1]):
    print(f"  {b:16s} {t/1000:8.2f} ms  ({100*t/total_kernel_us:5.1f}%)")

print("\n=== top 30 kernels by total time ===")
for name, t in sorted(kern_time.items(), key=lambda x: -x[1])[:30]:
    print(f"  {t/1000:8.2f} ms  x{kern_count[name]:5d}  {name[:95]}")

# --- GEMM shape attribution via CPU-side aten ops with Input Dims ---
# Under torch.compile many GEMMs are inside inductor kernels, but cuBLAS mm/addmm
# often still surface as aten ops with shapes. Aggregate by the 'characteristic'
# dimension so we can label MoE(1280/40960) vs prefix(4096) vs lm_head(100352).
shape_time = defaultdict(float)
shape_count = defaultdict(int)
for e in ev:
    if e.get("cat") in ("cpu_op", "user_annotation") and e.get("name", "").startswith("aten::"):
        nm = e["name"]
        if not any(g in nm for g in ("mm", "matmul", "addmm", "bmm", "linear")):
            continue
        args = e.get("args", {})
        dims = args.get("Input Dims") or args.get("Input dims") or args.get("input_dims")
        if dims:
            key = (nm, str(dims))
            shape_time[key] += e.get("dur", 0.0)
            shape_count[key] += 1
if shape_time:
    print("\n=== top matmul aten ops by CPU-scheduled time (shape-tagged) ===")
    for (nm, dims), t in sorted(shape_time.items(), key=lambda x: -x[1])[:25]:
        print(f"  {t/1000:8.2f} ms  x{shape_count[(nm,dims)]:5d}  {nm:16s} {dims[:70]}")
else:
    print("\n(no shape-tagged aten matmul ops — compiled into inductor kernels)")
