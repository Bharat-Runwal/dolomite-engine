#!/usr/bin/env python3
"""Does the active-parameter count agree with the real module tree, and is the reference right?

CPU only. Models are built on the META device -- no memory, no compute, no GPU. Mirrors
`test_sparse_w1w2_20260918.py` in structure.

FOUR SEPARATE CLAIMS:

  1. `base.py:206` currently reports active == total for EVERY energy arm (TEST 1). That is
     the bug; everything else is the fix.
  2. the live-module counters reproduce the Switch MoE's own `moe.py:418` answer on the one
     arm that already works, so the pattern is being followed and not reinvented (TEST 2).
  3. the STANDALONE YAML auditor agrees with the META build to the BYTE on all five published
     configs (TEST 3) -- that is what makes it usable before a launch.
  4. `layer_iterations` separates FLOPwt from ACTIVE, and the separation is the architectural
     claim about recurrence (TEST 4).

TEST 5 diagnoses the supplied reference values. TEST 6 checks the counters on the surrogate
head and on the w1w2 (two-matrix) expert kind, neither of which appears in any live config.
"""
import torch
import yaml

from lm_engine.hf_models.register_hf import register_model_classes

register_model_classes()
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import (  # noqa: E402
    build_boltzmann_moe,
)
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_paramcount import (  # noqa: E402
    audit_config,
    container_active_parameters,
    count_model_parameters,
    format_audit,
)
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_surrogate import (  # noqa: E402
    build_surrogate_boltzmann_moe,
)


ROOT = "/proj/dmfexp/nima/Code/dolomite-engine"

# name -> (path, reference TOTAL M, reference ACTIVE M, reference FLOPwt M or None)
CFGS = [
    ("cmix_134M_hybrid_32B_sparse", "configs/cmix/cmix_134M_hybrid_32B_sparse.yml", 125, 114, None),
    ("cmix_400M_hybrid_sparse",     "configs/cmix/cmix_400M_hybrid_sparse.yml",     375, 194, 276),
    ("cmix_400M_sandwich_sparse",   "configs/cmix/cmix_400M_sandwich_sparse.yml",   392, 148, 210),
    ("cmix_400M_baseline_switch",   "configs/cmix/cmix_400M_baseline_switch.yml",   375, 206, 206),
    ("cmix1B_12L_gptDense_32B",     "configs/cmix/cmix1B_12L_gptDense_32B.yml",     966, 244, None),
]


def meta_build(path):
    cfg = yaml.safe_load(open(f"{ROOT}/{path}"))["model_args"]["pretrained_config"]
    with torch.device("meta"):
        return AutoModelForCausalLM.from_config(config=AutoConfig.for_model(**cfg))


def base_py_active(model):
    """VERBATIM the recursion in model_wrapper/base.py:206, with nothing attached."""
    active = 0

    def rec(module):
        nonlocal active
        for m in module.children():
            if hasattr(m, "get_num_active_parameters"):
                active += m.get_num_active_parameters()
            else:
                for p in m.parameters(recurse=False):
                    active += p.numel()
                rec(m)

    rec(model)
    return active


print(__doc__.strip().split("\n")[0])
print("=" * 94)

models = {}
for name, path, *_ in CFGS:
    models[name] = meta_build(path)

# ---------------------------------------------------------------------------------------- #
print("\nTEST 1 -- the BUG, as it stands today. `base.py` active vs total, nothing patched.")
print(f"  {'config':32s} {'total':>12s} {'base.py active':>15s} {'ratio':>7s}  verdict")
for name, path, *_ in CFGS:
    m = models[name]
    tot = sum(p.numel() for p in m.parameters())
    act = base_py_active(m)
    bad = act == tot
    print(f"  {name:32s} {tot/1e6:11.3f}M {act/1e6:14.3f}M {act/tot:7.3f}  "
          f"{'*** EQUALS TOTAL -> WRONG' if bad else 'discounted (Switch MoE implements it)'}")

# ---------------------------------------------------------------------------------------- #
print("\nTEST 2 -- the counters reproduce moe.py:418 on the arm that ALREADY works,")
print("         and do not disturb it. The Switch baseline must be unchanged by the patch.")
m = models["cmix_400M_baseline_switch"]
before = base_py_active(m)
tot, after = count_model_parameters(m)
print(f"  base.py active before patch = {before/1e6:.3f}M")
print(f"  base.py active after  patch = {after/1e6:.3f}M   "
      f"{'UNCHANGED' if before == after else '*** CHANGED -- the patch touched the Switch arm'}")
sw = dict(m.named_modules())["transformer.h.6.mlp_block"]
print(f"  its MoE: all params {sum(p.numel() for p in sw.parameters())/1e6:.3f}M, "
      f"own get_num_active_parameters() = {sw.get_num_active_parameters()/1e6:.3f}M "
      f"(K={sw.num_experts}, top_k={sw.top_k})")

print("\n  and the TRUE active count for every arm, with the counters attached:")
truth = {}
for name, path, *_ in CFGS:
    mm = meta_build(path)
    t, a = count_model_parameters(mm)
    truth[name] = (t, a)
    print(f"    {name:32s} total {t/1e6:9.3f}M   active {a/1e6:9.3f}M   "
          f"({100*a/t:5.1f}% of total)")

# ---------------------------------------------------------------------------------------- #
print("\nTEST 3 -- the STANDALONE YAML auditor vs the META build, to the BYTE.")
print("         (this is the claim that matters for pre-launch auditing: no instantiation)")
audits = {}
ok_all = True
for name, path, *_ in CFGS:
    a = audit_config(f"{ROOT}/{path}")
    audits[name] = a
    t, act = truth[name]
    dt, da = a.total - t, a.active - act
    ok = dt == 0 and da == 0
    ok_all &= ok
    print(f"  {name:32s} total {a.total/1e6:9.3f}M (d={dt:+d})   "
          f"active {a.active/1e6:9.3f}M (d={da:+d})   {'EXACT' if ok else '*** MISMATCH'}")
    for w in a.warnings:
        print(f"      WARNING: {w}")
print(f"  all five exact: {ok_all}")

# ---------------------------------------------------------------------------------------- #
print("\nTEST 4 -- layer_iterations: FLOPwt is NOT active, and that is the recurrence claim.")
print(f"  {'config':32s} {'iters':>22s} {'TOTAL':>10s} {'ACTIVE':>10s} {'FLOPwt':>10s} {'FLOPwt-ex-emb':>14s}")
for name, path, *_ in CFGS:
    a = audits[name]
    it = "".join(str(b.iterations) for b in a.blocks)
    print(f"  {name:32s} {it:>22s} {a.total/1e6:9.3f}M {a.active/1e6:9.3f}M "
          f"{a.flop_weight/1e6:9.3f}M {a.flop_weight_ex_embedding/1e6:13.3f}M")
sw, sa, hy = (audits["cmix_400M_baseline_switch"], audits["cmix_400M_sandwich_sparse"],
              audits["cmix_400M_hybrid_sparse"])
print(f"\n  sandwich vs Switch baseline at 400M:")
print(f"    ACTIVE   {sa.active/1e6:.3f}M vs {sw.active/1e6:.3f}M  "
      f"-> {100*(sa.active/sw.active - 1):+.1f}% (the sandwich uses that much FEWER)")
print(f"    FLOPwt   {sa.flop_weight/1e6:.3f}M vs {sw.flop_weight/1e6:.3f}M  "
      f"-> {100*(sa.flop_weight/sw.flop_weight - 1):+.1f}%")
print(f"    FLOPwt ex-embedding {sa.flop_weight_ex_embedding/1e6:.3f}M vs "
      f"{sw.flop_weight_ex_embedding/1e6:.3f}M -> "
      f"{100*(sa.flop_weight_ex_embedding/sw.flop_weight_ex_embedding - 1):+.1f}%")
print("    NOTE: the supplied reference called these two ISO-FLOP at 210M vs 206M (+1.9%).")
print("    On the corrected counts the sandwich is BELOW the baseline in FLOP-weighted")
print("    parameters, so 'iso-FLOP' overstates the sandwich's compute. The recurrence claim")
print("    (far fewer active params at comparable compute) survives and gets stronger; the")
print("    specific phrase 'iso-FLOP with the Switch baseline' does not.")
print(f"\n  hybrid (1 energy block x6 of 7): ACTIVE {hy.active/1e6:.3f}M, "
      f"FLOPwt {hy.flop_weight/1e6:.3f}M -- recurrence buys COMPUTE, not sparsity, here.")

print("\n  per-block detail for the sandwich (the clearest case):")
print(format_audit("cmix_400M_sandwich_sparse", sa))

# ---------------------------------------------------------------------------------------- #
print("\nTEST 5 -- DIAGNOSING THE SUPPLIED REFERENCE VALUES.")
print("         Every reference TOTAL is LOW, by exactly n_dense_swiglu_blocks * d * I:")
print("         a swiglu MLP's c_fc is 2*intermediate_size wide (mlp.py:789), so a dense")
print("         swiglu block costs 3*d*I, not 2*d*I.")
print(f"  {'config':32s} {'mine':>10s} {'ref':>7s} {'gap':>10s} {'n*d*I':>11s} {'match':>7s}")
for name, path, rtot, ract, rflop in CFGS:
    a = audits[name]
    cfg = yaml.safe_load(open(f"{ROOT}/{path}"))["model_args"]["pretrained_config"]
    d = cfg["hidden_size"]
    n_dense = sum(1 for b in cfg["mlp_blocks"] if b.get("mlp_type", "MLP") == "MLP"
                  and str(b.get("activation_function", "")).lower() == "swiglu")
    I = next(b["intermediate_size"] for b in cfg["mlp_blocks"]
             if b.get("mlp_type", "MLP") == "MLP")
    gap = a.total - rtot * 1e6
    pred = n_dense * d * I
    print(f"  {name:32s} {a.total/1e6:9.3f}M {rtot:6d}M {gap/1e6:9.3f}M {pred/1e6:10.3f}M "
          f"{'YES' if abs(gap - pred) < 3.0e6 else 'no':>7s}")
print("\n  INDEPENDENT CHECK that mine is the right one: the task states the 1B's real")
print("  instantiated count from its safetensors header is 1002.1M.")
print(f"    mine (meta build) = {audits['cmix1B_12L_gptDense_32B'].total/1e6:.3f}M"
      f"   -> {100*abs(audits['cmix1B_12L_gptDense_32B'].total/1e6 - 1002.1)/1002.1:.3f}% off")
print(f"    reference         = 966M                 -> "
      f"{100*abs(966 - 1002.1)/1002.1:.2f}% off")
print("\n  The reference ACTIVE values equal reference TOTAL minus the CORRECT expert discount,")
print("  so the method was right and only the total was off:")
for name, path, rtot, ract, rflop in CFGS:
    a = audits[name]
    disc = a.total - a.active
    print(f"    {name:32s} ref {rtot}M - {disc/1e6:7.3f}M = {rtot - disc/1e6:8.3f}M "
          f"vs ref active {ract}M")
print("\n  Two more things the hand arithmetic could not have known:")
print("    * energy_attention is 2*d*d, not 4*d*d (Q and K only; V=K, no output proj), so the")
print("      4*d*d rule OVERCOUNTS every energy block by 2*d*d.")
print("    * every energy block also carries `proj` and `scale_ff`. With energy_proj_type:")
print("      psd_anti and the CommonConfig default energy_proj_rank=32 (not None), `proj` is")
print("      S(d x 32) + A(d x d):")
for name in ("cmix_134M_hybrid_32B_sparse", "cmix_400M_hybrid_sparse"):
    a = audits[name]
    eb = [b for b in a.blocks if b.kind == "energy"][0]
    print(f"        {name:32s} attn={eb.detail['attn']:,d}  "
          f"proj={eb.detail['proj']:,d} ({eb.detail['proj_type']})  "
          f"router={eb.detail.get('router', 0):,d}  scale_ff=1")

# ---------------------------------------------------------------------------------------- #
print("\nTEST 6 -- the two cases that appear in NO live config: w1w2 (TWO matrices per")
print("         expert) and the surrogate head (always-active router params).")
H, I, K, k = 64, 512, 8, 2
for kind, fused, n_mat in (("hopfield", True, 1), ("w1w2", False, 2)):
    torch.manual_seed(0)
    c = build_boltzmann_moe(expert_kind=kind, hidden_size=H, intermediate_size=I, n_experts=K,
                            top_k=k, temperature=1.0, initializer_range=0.05, m_width=1.0,
                            fused_experts=fused, e_sign_override="pos")
    tot = sum(p.numel() for p in c.parameters())
    act = container_active_parameters(c)
    exp = n_mat * I * H
    want = (exp * k) // K
    print(f"  kind={kind:8s} matrices={n_mat}  total={tot:8,d}  active={act:8,d}  "
          f"expected active={want:8,d}  {'OK' if act == want else '*** WRONG'}")

print("\n  with a rank-r proxy AND a surrogate head -- both ALWAYS active:")
torch.manual_seed(0)
c = build_boltzmann_moe(expert_kind="hopfield", hidden_size=H, intermediate_size=I, n_experts=K,
                        top_k=k, temperature=1.0, initializer_range=0.05, m_width=1.0,
                        fused_experts=True, e_sign_override="pos", proxy_rank=4,
                        proxy_kind="subspace", proxy_out_dim=16, proxy_loss_coef=0.01)
proxy_n = sum(p.numel() for p in c.moe.parameters())
print(f"    proxy only    : total={sum(p.numel() for p in c.parameters()):8,d}  "
      f"active={container_active_parameters(c):8,d}  (router {proxy_n:,d} all active)")
torch.manual_seed(0)
cs = build_surrogate_boltzmann_moe(
    expert_kind="hopfield", hidden_size=H, intermediate_size=I, n_experts=K, top_k=k,
    temperature=1.0, initializer_range=0.05, m_width=1.0, fused_experts=True,
    e_sign_override="pos", surrogate_coef=1.0, surrogate_kind="mlp", surrogate_hidden=32)
head_n = cs.moe.surrogate_parameter_numel()
print(f"    surrogate head: total={sum(p.numel() for p in cs.parameters()):8,d}  "
      f"active={container_active_parameters(cs):8,d}  (head {head_n:,d} all active)")
print(f"    head accounted for: "
      f"{container_active_parameters(cs) - (I * H * k) // K == head_n}")
print(f"    a naive K-leading test would have DISCOUNTED the router: proxy_V is (K, d, r) and")
print(f"    the fused weight is (K*I_e, d) -- `_expert_stacked` keys on K*I_e for that reason.")
