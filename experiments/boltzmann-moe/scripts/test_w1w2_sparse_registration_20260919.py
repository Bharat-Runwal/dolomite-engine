# **************************************************
# Registration check for the W1W2 sparse path -- 2026-09-19
# **************************************************
"""CPU-only validation for making ``BoltzmannMoEW1W2Sparse`` reachable from YAML.

TWO INDEPENDENT QUESTIONS, and they need different evidence:

  A (``--phase baseline`` then ``--phase check``)  NOTHING EXISTING CHANGED.
    Every ``mlp_blocks`` entry of every live config is built through ``get_mlp_block`` -- the
    real dispatch, not a hand-rolled builder -- and its float64 CPU forward is serialised to
    ``--store``. Run once BEFORE the edit (``baseline``) and once after (``check``); the
    comparison is BITWISE (``torch.equal`` on the raw bytes), not a tolerance. Six live jobs
    re-import this tree on preemption restart, so any nonzero diff is a launch blocker.
    Deterministic because every block is built after an explicit ``manual_seed``, so its
    parameters are a pure function of (seed, shapes, init_method) -- and because a float64 CPU
    GEMM is reproducible on a fixed host.

  B (``--phase new``)  THE NEW PATH WORKS. Builds the w1w2 + ``sparse_forward`` combination
    through ``get_mlp_block``, asserts the object is the w1w2 sparse class (a hopfield object
    here would mean the dispatch silently ignored ``expert_kind``), and checks the sparse
    output against the DENSE fused output with an ORACLE proxy, where the two are provably
    equal: at ``sparse_candidates = n_experts`` the candidate set is all K, so the only
    difference between the paths is the dispatch arithmetic.
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch
import yaml

# `--tree` picks WHICH COPY of lm_engine to import, which is how phase A removes the tree
# itself as the only variable: `git archive HEAD | tar -x` gives a pristine pre-edit checkout,
# and the two runs then differ in nothing but the code. Parsed before the imports below.
_ti = sys.argv.index("--tree") + 1 if "--tree" in sys.argv else 0
sys.path.insert(0, sys.argv[_ti] if _ti else "/proj/dmfexp/nima/Code/dolomite-engine")

import torch as _t
# PIN THE THREAD COUNT. A float64 CPU GEMM's reduction order depends on it, so an unpinned
# comparison reports last-bit diffs (~1e-16) on blocks the edit cannot touch -- MEASURED here:
# the same 50 blocks came out bitwise identical at 8 threads and differed at 1e-16 when only
# OPENBLAS_NUM_THREADS was exported. Determinism is a property of (code, seed, THREADS).
_t.set_num_threads(1)

from lm_engine.hf_models.models.energy.config import EnergyConfig                # noqa: E402
from lm_engine.hf_models.modeling_utils.mlp_blocks import get_mlp_block          # noqa: E402

REPO = "/proj/dmfexp/nima/Code/dolomite-engine"
CONFIGS = [
    f"{REPO}/configs/cmix/cmix_134M_hybrid_32B_sparse.yml",
    f"{REPO}/configs/cmix/cmix_400M_hybrid_sparse.yml",
    f"{REPO}/configs/cmix/cmix_400M_sandwich_sparse.yml",
    f"{REPO}/configs/cmix/cmix_400M_baseline_switch.yml",
    f"{REPO}/configs/cmix/cmix1B_12L_gptDense_32B.yml",
    f"{REPO}/configs/cmix/cmix_134M_hyb_w1w2_surrMLP_32B.yml",
    f"{REPO}/configs/iclr_26/ablations/abl_B_134M_6G1x6S.yml",
]
T = 8            # tokens; the point is bitwise reproducibility, not throughput


def build_block(cfg_dict: dict, layer_idx: int):
    cfg = EnergyConfig(**cfg_dict)
    torch.manual_seed(1234 + 17 * layer_idx)
    return get_mlp_block(cfg, False, layer_idx)


def forward_once(blk, hidden: int) -> torch.Tensor:
    blk = blk.double().eval()
    torch.manual_seed(999)
    x = torch.randn(1, T, hidden, dtype=torch.float64)
    with torch.no_grad():
        return blk(x).clone()


def phase_ab(store: str, compare: str | None) -> int:
    ref = torch.load(compare, weights_only=False) if compare else None
    out: dict[str, torch.Tensor] = {}
    bad = 0
    for path in CONFIGS:
        doc = yaml.safe_load(open(path))
        cfg_dict = doc["model_args"]["pretrained_config"]
        hidden = int(cfg_dict["hidden_size"])
        name = path.rsplit("/", 1)[-1]
        worst = 0.0
        nbit = 0
        for i in range(int(cfg_dict["num_layers"])):
            key = f"{name}:{i}"
            blk = build_block(cfg_dict, i)
            y = forward_once(blk, hidden)
            out[key] = y
            del blk
            if ref is not None:
                r = ref[key]
                same = (r.shape == y.shape and r.dtype == y.dtype
                        and torch.equal(r.view(torch.int64), y.view(torch.int64)))
                nbit += int(same)
                worst = max(worst, float((r - y).abs().max()) if r.shape == y.shape else float("inf"))
        if ref is None:
            print(f"  built + forwarded {int(cfg_dict['num_layers'])} blocks : {name}")
        else:
            L = int(cfg_dict["num_layers"])
            flag = "BITWISE IDENTICAL" if nbit == L else f"*** DIFFERS ({L - nbit}/{L}) ***"
            bad += L - nbit
            print(f"  {name:<42} blocks={L:<3} bitwise={nbit}/{L}  max|diff|={worst:.3e}  {flag}")
    torch.save(out, store)
    print(f"  -> wrote {len(out)} block outputs to {store}")
    return bad


# ------------------------------------------------------------------------------------------ #
# B: the new path
# ------------------------------------------------------------------------------------------ #
BASE_CFG = dict(
    model_type="energy", num_iterations=1, num_pre_layers=0, num_post_layers=0,
    layer_iterations=[1], initializer_range=0.05, layer_norm_epsilon=1e-5,
    normalization_function="rmsnorm", position_embedding_type="rope", rope_dim=32,
    hidden_size=64, num_layers=1, init_method="normal", tie_word_embeddings=True,
    energy_proj_type="psd_anti", bos_token_id=0, eos_token_id=0, pad_token_id=0,
    vocab_size=256, max_position_embeddings=128,
    sequence_mixer_blocks=[dict(sequence_mixer_type="energy_attention",
                                num_attention_heads=4, num_key_value_heads=4,
                                add_bias=False, attention_multiplier=0.125)],
)

W1W2_BLOCK = dict(
    mlp_type="EnergyFF_BoltzmannMoE", expert_kind="w1w2", intermediate_size=128, n_experts=8,
    top_k=2, routing_norm="zscore", temperature=1.0, repulsion_form="abs", repulsion_coef=0.0,
    n_repulsion_pairs=4, gelu_grad_method="sigmoid", e_sign_override="neg",
    activation_function="gelu", add_bias=False, renormalize_topk=True,
    repulsion_space="output", repulsion_subsample=8, repulsion_tensor_idx=True,
    fused_experts=True, sparse_forward=True, sparse_candidates=8, sparse_capacity_factor=8.0,
    proxy_rank=4, proxy_kind="subspace", proxy_out_dim=0, proxy_loss_coef=0.0,
    proxy_mu_convention="pre_mu", sinkhorn_iters=0, sinkhorn_persist_mu=False,
)


def mk(block: dict):
    c = dict(BASE_CFG)
    c["mlp_blocks"] = [block]
    cfg = EnergyConfig(**c)
    torch.manual_seed(7)
    return get_mlp_block(cfg, False, 0)


def phase_new() -> int:
    from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_w1w2_sparse import (
        BoltzmannMoEW1W2Sparse)
    from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import BoltzmannMoEFFEnergy
    bad = 0
    H = BASE_CFG["hidden_size"]
    torch.manual_seed(5)
    x = torch.randn(1, 16, H, dtype=torch.float64)

    def oracle(mod):
        """Exact w1w2 energies as the proxy -> the sparse nomination is the exact ranking."""
        def f(xx, _m=mod):
            W1, W2, W1v, W2v = _m._fused_views()
            z1 = torch.einsum("...d,kid->...ki", xx, W1v)
            z2 = torch.einsum("...d,kid->...ki", xx, W2v)
            return -(_m._expert_I ** -0.5) * (torch.nn.functional.gelu(z1) * z2).sum(-1)
        return f

    for tag, block, exp_cls in (
        ("w1w2 + sparse_forward (EnergyFF_BoltzmannMoE)", dict(W1W2_BLOCK), BoltzmannMoEW1W2Sparse),
        ("w1w2 + fused only     (EnergyFF_BoltzmannMoE)",
         {**W1W2_BLOCK, "sparse_forward": False, "proxy_rank": 0, "sparse_candidates": 0},
         BoltzmannMoEW1W2Sparse),
        ("hopfield + sparse (regression: must NOT be the w1w2 class)",
         {**W1W2_BLOCK, "expert_kind": "hopfield", "e_sign_override": "pos",
          "hopfield_grad_scale": "sqrt_consistent"}, BoltzmannMoEFFEnergy),
    ):
        c = mk(block).double().eval()
        moe = c.moe
        cls_ok = type(moe) is exp_cls
        print(f"  {tag}\n     class = {type(moe).__name__:<24} expected {exp_cls.__name__:<22}"
              f" {'OK' if cls_ok else '*** WRONG ***'}")
        bad += 0 if cls_ok else 1
        with torch.no_grad():
            y = moe(x)
        fin = bool(torch.isfinite(y).all())
        print(f"     forward finite = {fin}   |y|_max = {float(y.abs().max()):.4e}"
              f"   e_sign = {moe.e_sign}")
        bad += 0 if fin else 1
        if block.get("sparse_forward") and block["expert_kind"] == "w1w2":
            moe._proxy_energies = oracle(moe)
            with torch.no_grad():
                dense = moe._forward_fused(x).clone()
                sp = moe(x).clone()
            rel = float((sp - dense).norm() / dense.norm())
            ok = rel < 1e-14 and int(moe._sparse_overflow) == 0
            print(f"     ORACLE proxy, p=K: sparse vs dense rel err = {rel:.3e} "
                  f"overflow={int(moe._sparse_overflow)}  {'EXACT' if ok else '*** NOT EXACT ***'}")
            bad += 0 if ok else 1
        del c, moe

    # ---- w1w2 + sparse + SURROGATE selector ---- #
    surr = {**W1W2_BLOCK, "mlp_type": "EnergyFF_SurrogateBoltzmannMoE",
            "surrogate_kind": "mlp", "surrogate_hidden": 16, "surrogate_coef": 0.0,
            "use_surrogate": False, "surrogate_replaces_proxy": True,
            "surrogate_free_proxy": True, "proxy_loss_coef": 0.01, "proxy_mu_convention": "pre_mu",
            "sparse_explore": 1}
    c = mk(surr).double().eval()
    moe = c.moe
    print(f"  w1w2 + sparse + SURROGATE selector\n     class = {type(moe).__name__}"
          f"  is w1w2 sparse subclass = {isinstance(moe, BoltzmannMoEW1W2Sparse)}"
          f"  proxy freed = {moe._surr_proxy_freed}")
    bad += 0 if isinstance(moe, BoltzmannMoEW1W2Sparse) else 1
    with torch.no_grad():
        y = moe(x)
    print(f"     forward finite = {bool(torch.isfinite(y).all())}  |y|_max = "
          f"{float(y.abs().max()):.4e}")
    bad += 0 if bool(torch.isfinite(y).all()) else 1
    moe.surrogate_energies = oracle(moe)
    with torch.no_grad():
        dense = moe._forward_fused(x).clone()
        sp = moe(x).clone()
    rel = float((sp - dense).norm() / dense.norm())
    ok = rel < 1e-14
    print(f"     ORACLE head, p=K: sparse vs dense rel err = {rel:.3e}  "
          f"{'EXACT' if ok else '*** NOT EXACT ***'}")
    bad += 0 if ok else 1
    return bad


# ------------------------------------------------------------------------------------------ #
# PRE-FLIGHT RULE 7, BOTH DIRECTIONS.
#   forward  : every field of _EnergyFFBoltzmannMoEArgs reaches the builder with its VALUE
#              (a field that parses onto the pydantic object and is never named in
#              get_mlp_block is a SILENT no-op -- it cannot be found by reading the YAML)
#   backward : every parameter of the builder is either fed by get_mlp_block or is a
#              deliberate default (an unfed parameter is a knob no config can set)
#   resolve  : the value is then READ BACK OFF THE BUILT `.moe`, not off the container --
#              CLAUDE.md pre-flight 7 records that a probe reading the FusedMoEContainer gets
#              None for everything and its fallback branch reports whatever you told it to.
# ------------------------------------------------------------------------------------------ #
# Deliberately NOT forwarded, with the reason. Anything else missing is a bug.
NOT_FORWARDED = {
    "mlp_type": "the dispatch key itself",
    "activation_function": "unused by the composable energy FF (gelu is fixed); the frozen "
                           "builder does not take it either",
    "dropout": "not implemented by the composable family (the legacy class had it)",
}
# Builder parameters that get_mlp_block deliberately leaves at their default.
UNFED_OK = {"hopfield_grad_scale": "dropped for w1w2: a Hopfield-only gradient prefactor"}

# One distinguishable, legal value per field.
PROBE = dict(
    mlp_type="EnergyFF_BoltzmannMoE",
    intermediate_size=128, n_experts=8, expert_kind="w1w2", temperature=0.75,
    repulsion_coef=0.125, n_repulsion_pairs=6, repulsion_form="hinge", routing_norm="sqrt_width",
    hopfield_grad_scale="inv_sqrt", top_k=3, renormalize_topk=True, repulsion_interval=7,
    repulsion_scale_comp=False, fused_experts=True, proxy_rank=5, proxy_loss_coef=0.031,
    proxy_route=True, proxy_kind="subspace", proxy_init="svd", proxy_out_dim=11, proxy_iters=1,
    proxy_mu_convention="pre_mu", cos_probe_interval=13, cos_probe_pairs=9,
    repulsion_space="weight", e_sign_override="neg", sinkhorn_iters=3, sinkhorn_persist_mu=True,
    sinkhorn_mu_iters=2, repulsion_tensor_idx=True, sparse_backproj=False, sparse_forward=True,
    sparse_candidates=6, sparse_explore=1, sparse_start_step=250, repulsion_subsample=17,
    sparse_capacity_factor=2.5, track_load=False, balance_rate=0.0,
    gelu_grad_method="erf_exact", activation_function="gelu", dropout=0.0, add_bias=False,
)


def phase_fields() -> int:
    import inspect
    from lm_engine.hf_models.config.mlp import _EnergyFFBoltzmannMoEArgs
    from lm_engine.hf_models.modeling_utils import mlp_blocks as MB
    from lm_engine.hf_models.modeling_utils.mlp_blocks import energy_ff_w1w2_sparse as WS
    from lm_engine.hf_models.modeling_utils.mlp_blocks import energy_ff as EF

    fields = set(_EnergyFFBoltzmannMoEArgs.model_fields)
    missing_probe = fields - set(PROBE)
    assert not missing_probe, f"PROBE is out of date, add: {sorted(missing_probe)}"
    print(f"  _EnergyFFBoltzmannMoEArgs has {len(fields)} fields")

    captured: dict[str, dict] = {}

    def spy(name):
        def f(**kw):
            captured[name] = kw
            return "SENTINEL"
        return f

    real_bm, real_ws = MB.build_boltzmann_moe, WS.build_boltzmann_moe_w1w2_sparse
    MB.build_boltzmann_moe, WS.build_boltzmann_moe_w1w2_sparse = spy("bm"), spy("ws")
    try:
        for tag, over in (("w1w2+fused -> w1w2 sparse builder", {}),
                          ("hopfield -> frozen builder", dict(expert_kind="hopfield"))):
            captured.clear()
            blk = {**PROBE, "mlp_type": "EnergyFF_BoltzmannMoE", **over}
            c = dict(BASE_CFG)
            c["hidden_size"] = 64
            c["mlp_blocks"] = [blk]
            got = get_mlp_block(EnergyConfig(**c), False, 0)
            which = "ws" if not over else "bm"
            ok = got == "SENTINEL" and which in captured
            print(f"  {tag:<38} builder reached = {list(captured)} {'OK' if ok else '*** NO ***'}")
            if not ok:
                return 1
        kw_ws, kw_bm = captured.get("ws"), captured["bm"]
    finally:
        MB.build_boltzmann_moe, WS.build_boltzmann_moe_w1w2_sparse = real_bm, real_ws

    # re-capture the w1w2 call (the loop above clears)
    MB.build_boltzmann_moe, WS.build_boltzmann_moe_w1w2_sparse = spy("bm"), spy("ws")
    try:
        captured.clear()
        c = dict(BASE_CFG); c["hidden_size"] = 64
        c["mlp_blocks"] = [{**PROBE, "mlp_type": "EnergyFF_BoltzmannMoE"}]
        get_mlp_block(EnergyConfig(**c), False, 0)
        kw_ws = captured["ws"]
    finally:
        MB.build_boltzmann_moe, WS.build_boltzmann_moe_w1w2_sparse = real_bm, real_ws

    bad = 0
    print("\n  FORWARD -- every pydantic field reaches a builder with its value:")
    for fld in sorted(fields):
        want = PROBE[fld]
        in_ws, in_bm = fld in kw_ws, fld in kw_bm
        if fld in NOT_FORWARDED:
            print(f"    {fld:<28} SKIPPED ON PURPOSE  ({NOT_FORWARDED[fld]})")
            continue
        if fld == "hopfield_grad_scale":
            okv = (not in_ws) and kw_bm.get(fld) == want
            print(f"    {fld:<28} frozen builder = {kw_bm.get(fld)!r}  w1w2 builder = "
                  f"{'(dropped, correct)' if not in_ws else '*** FORWARDED, WOULD TypeError ***'}"
                  f"  {'OK' if okv else '*** BAD ***'}")
            bad += 0 if okv else 1
            continue
        if fld == "expert_kind":
            okv = (not in_ws) and kw_bm.get(fld) == "hopfield"
            print(f"    {fld:<28} implied by the w1w2 builder "
                  f"{'(dropped, correct)' if not in_ws else '*** FORWARDED ***'}"
                  f"  {'OK' if okv else '*** BAD ***'}")
            bad += 0 if okv else 1
            continue
        okv = in_ws and in_bm and kw_ws[fld] == want and kw_bm[fld] == want
        if not okv:
            bad += 1
        print(f"    {fld:<28} ws={kw_ws.get(fld, '<ABSENT>')!r:<12} "
              f"bm={kw_bm.get(fld, '<ABSENT>')!r:<12} want={want!r:<12} "
              f"{'OK' if okv else '*** SILENT NO-OP ***'}")

    print("\n  BACKWARD -- every builder parameter is fed or a deliberate default:")
    for name, fn in (("build_boltzmann_moe", real_bm),
                     ("build_boltzmann_moe_w1w2_sparse", real_ws)):
        sig = inspect.signature(fn)
        pars = {p for p, v in sig.parameters.items()
                if v.kind is not inspect.Parameter.VAR_KEYWORD}
        fed = set(kw_bm if name.endswith("moe") else kw_ws)
        unfed = pars - fed - {"layer_idx", "hidden_size", "init_method", "initializer_range",
                              "m_width"}
        strays = sorted(unfed - set(UNFED_OK))
        print(f"    {name:<32} {len(pars)} named params, unfed = "
              f"{sorted(unfed) or 'none'}  {'OK' if not strays else '*** UNREACHABLE KNOB ***'}")
        bad += len(strays)
    # the w1w2 builder swallows the rest through **moe_kwargs -> BoltzmannMoEW1W2Sparse ->
    # BoltzmannMoEFFEnergy; check every forwarded key is actually accepted there
    accepted = set()
    for cls in (WS.BoltzmannMoEW1W2Sparse, EF.BoltzmannMoEFFEnergy):
        accepted |= set(inspect.signature(cls.__init__).parameters)
    swallowed = set(kw_ws) - set(inspect.signature(real_ws).parameters)
    rej = sorted(swallowed - accepted)
    print(f"    **moe_kwargs pass-through: {len(swallowed)} keys, not accepted downstream = "
          f"{rej or 'none'}  {'OK' if not rej else '*** TypeError AT BUILD ***'}")
    bad += len(rej)

    print("\n  RESOLVE -- read back off the BUILT .moe (not the container):")
    real = dict(PROBE)
    real.update(repulsion_space="output", repulsion_coef=0.0, proxy_init="random",
                proxy_out_dim=0, sparse_start_step=0, cos_probe_interval=0)
    c = dict(BASE_CFG); c["hidden_size"] = 64
    c["mlp_blocks"] = [{**real, "mlp_type": "EnergyFF_BoltzmannMoE"}]
    torch.manual_seed(3)
    built = get_mlp_block(EnergyConfig(**c), False, 0)
    moe = built.moe
    checks = [("temperature", moe.temperature, real["temperature"]),
              ("top_k", int(moe.top_k), real["top_k"]),
              ("e_sign", moe.e_sign, "neg"),
              ("routing_norm", moe.routing_norm, real["routing_norm"]),
              ("renormalize_topk", moe.renormalize_topk, True),
              ("fused_experts", moe.fused_experts, True),
              ("fused_spec['kind']", moe._fused_spec["kind"], "w1w2"),
              ("sparse_forward", moe.sparse_forward, True),
              ("sparse_candidates", moe.sparse_candidates, real["sparse_candidates"]),
              ("sparse_explore", moe.sparse_explore, real["sparse_explore"]),
              ("sparse_capacity_factor", moe.sparse_capacity_factor, real["sparse_capacity_factor"]),
              ("proxy_rank", moe.proxy_rank, real["proxy_rank"]),
              ("proxy_kind", moe.proxy_kind, real["proxy_kind"]),
              ("proxy_loss_coef", moe.proxy_loss_coef, real["proxy_loss_coef"]),
              ("proxy_mu_convention", moe.proxy_mu_convention, "pre_mu"),
              ("proxy_route", moe.proxy_route, True),
              ("sinkhorn_iters", moe.sinkhorn_iters, real["sinkhorn_iters"]),
              ("sinkhorn_persist_mu", moe.sinkhorn_persist_mu, True),
              ("sinkhorn_mu_iters", moe.sinkhorn_mu_iters, real["sinkhorn_mu_iters"]),
              ("repulsion_tensor_idx", moe.repulsion_tensor_idx, True),
              ("repulsion_interval", moe.repulsion_interval, real["repulsion_interval"]),
              ("repulsion_form", moe.repulsion_form, real["repulsion_form"]),
              ("repulsion_subsample", moe.repulsion_subsample, real["repulsion_subsample"]),
              ("track_load", moe.track_load, False),
              ("gelu_grad_method", moe.gelu_grad_method, real["gelu_grad_method"]),
              ("n_repulsion_pairs", moe.n_repulsion_pairs, real["n_repulsion_pairs"]),
              ("class", type(moe).__name__, "BoltzmannMoEW1W2Sparse")]
    for nm, got_v, want_v in checks:
        okv = got_v == want_v
        bad += 0 if okv else 1
        print(f"    {nm:<26} = {got_v!r:<26} want {want_v!r:<20} "
              f"{'OK' if okv else '*** DID NOT REACH THE MODEL ***'}")

    # e_sign default must NOT depend on fused_experts
    print("\n  SIGN NEUTRALITY -- fused_experts must not change the routing sign:")
    for fe in (False, True):
        c = dict(BASE_CFG); c["hidden_size"] = 64
        b = {**real, "mlp_type": "EnergyFF_BoltzmannMoE", "fused_experts": fe,
             "sparse_forward": False, "proxy_rank": 0, "sparse_candidates": 0,
             "sparse_explore": 0}
        b.pop("e_sign_override")
        c["mlp_blocks"] = [b]
        torch.manual_seed(3)
        m2 = get_mlp_block(EnergyConfig(**c), False, 0).moe
        print(f"    fused_experts={str(fe):<5} e_sign = {m2.e_sign!r}  class = {type(m2).__name__}")
        bad += 0 if m2.e_sign == "pos" else 1
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("baseline", "check", "new", "fields"))
    ap.add_argument("--store", default="/tmp/w1w2reg/blocks.pt")
    ap.add_argument("--compare", default=None)
    ap.add_argument("--tree", default="/proj/dmfexp/nima/Code/dolomite-engine")
    a = ap.parse_args()
    logging.basicConfig(level=logging.ERROR)
    torch.use_deterministic_algorithms(True)
    if a.phase == "new":
        n = phase_new()
    elif a.phase == "fields":
        n = phase_fields()
    else:
        n = phase_ab(a.store, a.compare)
    print(("\nALL OK" if n == 0 else f"\n*** {n} FAILURE(S) ***"))
    sys.exit(1 if n else 0)