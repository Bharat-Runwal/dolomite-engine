#!/usr/bin/env python3
"""Did wiring the param counters / dominant-expert metric / proxy_mu_convention / surrogate
mlp_type change ANY live arm?  CPU only, no GPU, no bsub.

Two independent jobs, selected by argv:

  ``--dump DIR``   build the five LIVE configs on CPU and save every forward output to DIR.
                   Imports NOTHING new, so the SAME file runs against the pre-edit package
                   (``PYTHONPATH=/tmp/base_lm``) and against the edited one.  Three forwards
                   per config: eval, train, and train with the sparse path forced active --
                   the last is the one that exercises ``_forward_sparse`` (every live energy
                   arm has ``sparse_start_step`` > 0, so a fresh model is in its DENSE phase).
  ``--cmp A B``    bitwise-compare two dump dirs.
  ``--new``        exercise the NEW things: true active-parameter counts through
                   ``calculate_num_parameters``'s own recursion, ``load_n_dominant_experts``
                   under torch.compile, and an ``EnergyFF_SurrogateBoltzmannMoE`` config
                   resolving end to end.

``vocab_size`` is shrunk to 1024 in the dump so five models fit in RAM quickly; it touches no
MoE code path.  Everything else is verbatim from the published YAML.
"""
import os
import random
import sys

import torch
import yaml

ROOT = "/proj/dmfexp/nima/Code/dolomite-engine"
CFGS = [
    ("cmix_134M_hybrid_32B_sparse", "configs/cmix/cmix_134M_hybrid_32B_sparse.yml"),
    ("cmix_400M_hybrid_sparse",     "configs/cmix/cmix_400M_hybrid_sparse.yml"),
    ("cmix_400M_sandwich_sparse",   "configs/cmix/cmix_400M_sandwich_sparse.yml"),
    ("cmix_400M_baseline_switch",   "configs/cmix/cmix_400M_baseline_switch.yml"),
    ("cmix1B_12L_gptDense_32B",     "configs/cmix/cmix1B_12L_gptDense_32B.yml"),
]
B, S, VOCAB = 2, 16, 1024


def _hf():
    from lm_engine.hf_models.register_hf import register_model_classes
    register_model_classes()
    from transformers import AutoConfig, AutoModelForCausalLM
    return AutoConfig, AutoModelForCausalLM


def raw_cfg(path, **over):
    c = yaml.safe_load(open(f"{ROOT}/{path}"))["model_args"]["pretrained_config"]
    c["vocab_size"] = VOCAB
    c.update(over)
    return c


def seed_all(s=1234):
    torch.manual_seed(s)
    random.seed(s)


def build(cfg_dict):
    AutoConfig, AutoModelForCausalLM = _hf()
    seed_all()
    return AutoModelForCausalLM.from_config(config=AutoConfig.for_model(**cfg_dict))


def one_forward(model, mode, force_sparse):
    """mode in {'eval','train'}; returns (logits, aux_loss_float)."""
    from lm_engine.hf_models.loss import clear_aux_loss, get_aux_loss
    from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import BoltzmannMoEFFEnergy

    model.train(mode == "train")
    for m in model.modules():
        if isinstance(m, BoltzmannMoEFFEnergy) and hasattr(m, "set_training_step"):
            # Flipping the dense->sparse gate also fires `_svd_refit_proxy`, i.e. one SVD per
            # expert (256 of them on the 1B) -- minutes of CPU per config, in code this task
            # does not touch, and LAPACK is not a determinism guarantee across processes.
            # Mark it done so the gate flips and the SPARSE FORWARD still runs, on the random
            # proxy init. Identical on both sides of the A/B.
            if force_sparse:
                m._svd_done = True
            m.set_training_step(10 ** 7 if force_sparse else 1)
    seed_all(99)
    ids = torch.randint(0, VOCAB, (B, S))
    clear_aux_loss()
    seed_all(7)                      # repulsion pair draw + any in-forward RNG
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    try:
        aux = float(get_aux_loss())
    except Exception:
        aux = float("nan")
    clear_aux_loss()
    return logits.detach().clone(), aux


def dump(outdir):
    os.makedirs(outdir, exist_ok=True)
    for name, path in CFGS:
        model = build(raw_cfg(path))
        rec = {}
        for tag, mode, sp in (("eval", "eval", False), ("train", "train", False),
                              ("train_sparse", "train", True)):
            lg, aux = one_forward(model, mode, sp)
            rec[tag] = {"logits": lg, "aux": aux}
            print(f"  {name:32s} {tag:12s} sum={lg.double().sum().item():.10e} aux={aux}")
        rec["n_params"] = sum(p.numel() for p in model.parameters())
        torch.save(rec, f"{outdir}/{name}.pt")
        del model
    print(f"dumped to {outdir}")


def cmp(a, b):
    ok = True
    print(f"  {'config':32s} {'forward':12s} {'max|diff|':>12s}  {'aux diff':>12s}  verdict")
    for name, _ in CFGS:
        ra, rb = torch.load(f"{a}/{name}.pt"), torch.load(f"{b}/{name}.pt")
        assert ra["n_params"] == rb["n_params"], (name, ra["n_params"], rb["n_params"])
        for tag in ("eval", "train", "train_sparse"):
            d = (ra[tag]["logits"].double() - rb[tag]["logits"].double()).abs().max().item()
            da = abs(ra[tag]["aux"] - rb[tag]["aux"]) if ra[tag]["aux"] == ra[tag]["aux"] else 0.0
            good = (d == 0.0) and (da == 0.0)
            ok &= good
            print(f"  {name:32s} {tag:12s} {d:12.3e}  {da:12.3e}  "
                  f"{'BITWISE IDENTICAL' if good else '*** DIFFERS'}")
    print("\nVALIDATION A:", "PASS -- every live config bitwise unchanged" if ok else "*** FAIL")
    return 0 if ok else 1


# ------------------------------------------------------------------------------------ #
def new_checks():
    from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import (
        BoltzmannMoEFFEnergy, FusedMoEContainer)

    print("\nB1 -- calculate_num_parameters()'s OWN recursion, verbatim, on a meta build.")
    EXPECT = {"cmix_134M_hybrid_32B_sparse": 123.243, "cmix_400M_hybrid_sparse": 219.427,
              "cmix_400M_sandwich_sparse": 156.539, "cmix_400M_baseline_switch": 231.697,
              "cmix1B_12L_gptDense_32B": 279.320}

    def base_py(model):                     # verbatim model_wrapper/base.py:206
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

    AutoConfig, AutoModelForCausalLM = _hf()
    ok = True
    print(f"  {'config':32s} {'total':>12s} {'active':>12s} {'expected':>12s}  verdict")
    for name, path in CFGS:
        c = yaml.safe_load(open(f"{ROOT}/{path}"))["model_args"]["pretrained_config"]
        with torch.device("meta"):
            m = AutoModelForCausalLM.from_config(config=AutoConfig.for_model(**c))
        tot = sum(p.numel() for p in m.parameters())
        act = base_py(m)
        exp = EXPECT[name]
        good = abs(act / 1e6 - exp) < 0.001
        ok &= good
        print(f"  {name:32s} {tot/1e6:11.3f}M {act/1e6:11.3f}M {exp:11.3f}M  "
              f"{'OK' if good else '*** MISMATCH'}")
        del m

    print("\nB2 -- load_n_dominant_experts / load_n_used_experts under torch.compile.")
    from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe
    torch.manual_seed(0)
    c = build_boltzmann_moe(expert_kind="hopfield", hidden_size=32, intermediate_size=128,
                            n_experts=8, top_k=2, initializer_range=0.05, m_width=1.0,
                            fused_experts=True, e_sign_override="pos", routing_norm="zscore",
                            renormalize_topk=True, sinkhorn_iters=3, proxy_rank=4,
                            proxy_kind="subspace", proxy_loss_coef=0.01)
    c.train()
    x = torch.randn(4, 16, 32)
    import torch._dynamo as dynamo
    dynamo.reset()
    dynamo.utils.counters.clear()
    compiled = torch.compile(c, fullgraph=False)
    from lm_engine.hf_models.loss import clear_aux_loss
    clear_aux_loss()
    compiled(x)
    clear_aux_loss()
    breaks = sum(dynamo.utils.counters["graph_break"].values())
    print(f"    dynamo: {breaks} graph break(s) "
          f"{'(pre-existing: repulsion / add_aux_loss)' if breaks else ''}")
    print(f"    torch.compiler.is_compiling() was traced -> _log_metrics gives "
          f"{c.get_metrics()}")
    m = c.pop_load_metrics()
    for key in ("load_n_dominant_experts", "load_n_used_experts"):
        print(f"    {key:26s} = {m.get(key)}   {'OK' if key in m else '*** MISSING'}")
        ok &= key in m
    print(f"    (other keys present: {sorted(k for k in m if k.startswith('load_'))})")
    # eager must agree with compiled on the same input
    torch.manual_seed(0)
    c2 = build_boltzmann_moe(expert_kind="hopfield", hidden_size=32, intermediate_size=128,
                             n_experts=8, top_k=2, initializer_range=0.05, m_width=1.0,
                             fused_experts=True, e_sign_override="pos", routing_norm="zscore",
                             renormalize_topk=True, sinkhorn_iters=3, proxy_rank=4,
                             proxy_kind="subspace", proxy_loss_coef=0.01)
    c2.train()
    clear_aux_loss()
    c2(x)
    clear_aux_loss()
    m2 = c2.pop_load_metrics()
    same = (m["load_n_dominant_experts"] == m2["load_n_dominant_experts"])
    ok &= same
    print(f"    compiled {m['load_n_dominant_experts']} vs eager {m2['load_n_dominant_experts']}"
          f"  -> {'AGREE' if same else '*** DISAGREE'}")

    print("\nB3 -- an EnergyFF_SurrogateBoltzmannMoE config resolves END TO END.")
    from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_surrogate import (
        SurrogateBoltzmannMoEFFEnergy)
    cfg = raw_cfg(CFGS[0][1])
    blk = dict(cfg["mlp_blocks"][6])
    blk["mlp_type"] = "EnergyFF_SurrogateBoltzmannMoE"
    blk["sparse_forward"] = False          # the mixin refuses it, on purpose
    blk.pop("sparse_candidates", None); blk.pop("sparse_explore", None)
    blk.pop("sparse_start_step", None)
    blk.update(surrogate_coef=0.05, use_surrogate=False, surrogate_kind="mlp",
               surrogate_hidden=64, surrogate_track_agree=True)
    cfg["mlp_blocks"] = cfg["mlp_blocks"][:6] + [blk]
    model = build(cfg)
    ffwd = dict(model.named_modules())["transformer.h.6.ffwd"]
    moe = ffwd.moe
    checks = [
        ("container is FusedMoEContainer", isinstance(ffwd, FusedMoEContainer)),
        ("moe is SurrogateBoltzmannMoEFFEnergy", isinstance(moe, SurrogateBoltzmannMoEFFEnergy)),
        ("surrogate_W1 exists", getattr(moe, "surrogate_W1", None) is not None),
        ("surrogate_W2 exists (kind=mlp)", getattr(moe, "surrogate_W2", None) is not None),
        ("surrogate_W1 shape (h,H)", tuple(moe.surrogate_W1.shape) == (64, 768)),
        ("surrogate_W2 shape (K,h)", tuple(moe.surrogate_W2.shape) == (16, 64)),
        ("surrogate_coef reached the module", moe.surrogate_coef == 0.05),
        ("surrogate_kind reached the module", moe.surrogate_kind == "mlp"),
        ("expert knobs still reached it (e_sign)", moe.e_sign == "pos"),
        ("sinkhorn_iters reached it", moe.sinkhorn_iters == 3),
        ("proxy_rank reached it", moe.proxy_rank == 16),
        ("renormalize_topk reached it", moe.renormalize_topk is True),
        ("repulsion_subsample reached it", moe.repulsion_subsample == 64),
        ("proxy_mu_convention default", moe.proxy_mu_convention == "legacy"),
    ]
    for label, good in checks:
        ok &= bool(good)
        print(f"    {label:42s} {'OK' if good else '*** FAIL'}")
    lg, aux = one_forward(model, "train", False)
    print(f"    train forward ran: logits {tuple(lg.shape)} aux={aux:.6f}")

    print("\nB4 -- proxy_mu_convention: 'pre_mu' changes the DISTILLATION TARGET and nothing else.")
    from lm_engine.hf_models.loss import clear_aux_loss, get_aux_loss

    def run(conv):
        torch.manual_seed(0)
        cc = build_boltzmann_moe(expert_kind="hopfield", hidden_size=32, intermediate_size=128,
                                 n_experts=8, top_k=2, initializer_range=0.05, m_width=1.0,
                                 fused_experts=True, e_sign_override="pos",
                                 routing_norm="zscore", renormalize_topk=True,
                                 sinkhorn_iters=3, proxy_rank=4, proxy_kind="subspace",
                                 proxy_loss_coef=0.01, proxy_mu_convention=conv)
        cc.double(); cc.train()
        torch.manual_seed(5)
        xx = torch.randn(4, 16, 32, dtype=torch.float64)
        clear_aux_loss()
        out = cc(xx)
        a = float(get_aux_loss()); clear_aux_loss()
        return out, a, cc.pop_load_metrics()

    o_l, a_l, m_l = run("legacy")
    o_p, a_p, m_p = run("pre_mu")
    dout = (o_l - o_p).abs().max().item()
    print(f"    output max|diff|      = {dout:.3e}   "
          f"{'OK (routing untouched)' if dout == 0.0 else '*** the forward moved'}")
    print(f"    aux (KL) legacy       = {a_l:.8f}")
    print(f"    aux (KL) pre_mu       = {a_p:.8f}   "
          f"{'OK (target moved)' if a_l != a_p else '*** target did NOT move'}")
    print(f"    proxy_topk_agree      = {m_l.get('proxy_topk_agree')} -> "
          f"{m_p.get('proxy_topk_agree')}")
    ok &= (dout == 0.0) and (a_l != a_p)

    print("\nVALIDATION B:", "PASS" if ok else "*** FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "--dump":
        dump(sys.argv[2]); sys.exit(0)
    if sys.argv[1] == "--cmp":
        sys.exit(cmp(sys.argv[2], sys.argv[3]))
    if sys.argv[1] == "--new":
        sys.exit(new_checks())
    raise SystemExit(__doc__)
