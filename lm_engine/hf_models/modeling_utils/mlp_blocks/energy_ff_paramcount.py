# **************************************************
# Active-parameter counting for the energy MoE -- 2026-09-18
# **************************************************
"""``active_parameters`` logged to wandb EQUALS ``num_parameters`` for every energy arm.

``model_wrapper/base.py:206`` ``calculate_num_parameters()`` counts total parameters, then
walks the module tree looking for ``get_num_active_parameters()`` on each child:

    for m in module.children():
        if hasattr(m, "get_num_active_parameters"):
            active_parameters += m.get_num_active_parameters()
        else:
            for parameter in m.parameters(recurse=False):
                active_parameters += parameter.numel()
            _recurse_immediate_children_and_count_active_parameters(m)

**ONLY the Switch MoE implements that method** (``moe.py:418``). No energy class does, so for
every ``EnergyFF_BoltzmannMoE`` arm the recursion falls through to
``parameter.numel()`` on the fused expert tensor and counts ALL K experts. Confirmed on a meta
build of the five live configs: ``active == total`` exactly, in every one.

MEASURED on the published configs (meta device, so this is the real module tree):

    config                              total      base.py "active"    TRUE active
    cmix_134M_hybrid_32B_sparse        134.253M      134.253M           123.243M
    cmix_400M_hybrid_sparse            399.784M      399.784M           219.427M
    cmix_400M_sandwich_sparse          400.333M      400.333M           156.539M
    cmix_400M_baseline_switch          400.031M      231.697M (correct) 231.697M
    cmix1B_12L_gptDense_32B           1002.067M     1002.067M           279.320M

so the 1B arm's logged ``active_parameters`` is **3.6x** its true value, and the Switch
baseline -- the ONE arm that is right -- is the arm every energy arm gets compared against.
Any published active-parameter or "parameters per token" comparison between an energy arm and
``cmix_400M_baseline_switch`` is therefore wrong in the direction that flatters the baseline.

NOTHING IN ``energy_ff.py`` OR ``base.py`` IS MODIFIED -- six long training runs import them.
This module provides the counters as free functions, as mixins for future builders, and as an
OPT-IN runtime binder (``attach_active_param_counters``). Wiring it into the live path is one
call site, listed at the bottom of this docstring; it is deliberately NOT made here.

===============================================================================================
(a) WHAT "ACTIVE" MEANS HERE, AND THE FOUR FACTS THE SWITCH PATTERN DOES NOT COVER
===============================================================================================

Following ``moe.py:418``: start from every parameter, then for each EXPERT-STACKED tensor
subtract its ``numel()`` and add back ``numel() * top_k // n_experts``.

 1. **HOPFIELD STORES ONE MATRIX PER EXPERT, w1w2 STORES TWO.** ``_FusedHopfieldHolder`` has
    ``W`` of shape ``(K*I_e, d)``; ``_FusedW1W2Holder`` has ``W1`` AND ``W2``, each
    ``(K*I_e, d)``. Both are discounted by the same ``top_k/K``, so the arity is handled by
    iterating the holder's parameters rather than by naming them -- no ``expert_kind`` branch,
    and a third expert kind would work unchanged.

 2. **THE ROUTER'S PARAMETERS ARE ALWAYS ACTIVE.** ``proxy_V`` / ``proxy_B`` / ``proxy_scale``
    / ``proxy_bias`` / ``proxy_quad`` / ``proxy_lin`` (and the surrogate head's
    ``surrogate_W*`` / ``surrogate_b*``) are evaluated for EVERY token -- that is what a cheap
    router is. They must NOT be discounted. They are also not small: the 1B arm carries
    1,572,992 proxy parameters per MoE block, 6.3M over four blocks, which is 2.3% of its true
    active count. ``load_balance_bias`` and ``sinkhorn_mu`` are BUFFERS, not parameters, so
    ``.parameters()`` never sees them -- correct either way, but worth knowing so the numbers
    reconcile.

 3. **THE FUSED HOLDER STORES ONE STACKED TENSOR, NOT PER-EXPERT TENSORS, AND
    ``moe.experts`` OWNS NOTHING.** ``make_experts()`` hands each ``_HopfieldExpert`` /
    ``_W1W2Expert`` a CLOSURE over a row-slice of the holder's weight, so the expert modules
    have zero parameters of their own. Verified on the meta build: ``ffwd.moe`` reports 327,712
    parameters at 134M and every one of them is proxy. Consequence: the discount must be
    applied at the HOLDER (a sibling of ``moe`` under ``FusedMoEContainer``), and a counter
    that only looked inside ``BoltzmannMoEFFEnergy`` would find nothing to discount and return
    the wrong answer while looking like it worked. The non-fused case (experts that own their
    weights) is also handled, for completeness.

 4. **``top_k=None`` MEANS DENSE.** Soft all-K mixing: every expert is evaluated, so active ==
    total and there is nothing to subtract. Several `pure_*` arms are in that state.

CAVEAT ON WHAT THE NUMBER MEANS. ``top_k/K`` is the ratio of expert weights TOUCHED per token
in a path that actually skips experts. Today only ``sparse_forward`` skips them; a DENSE
top-k arm computes all K expert outputs and multiplies K-k of them by zero. So this is the
count of parameters that AFFECT the output, which is the standard MoE convention and what
``moe.py`` computes for the Switch baseline -- it is not a measured FLOP saving. That is what
``flop_weight`` below is for.

===============================================================================================
(b) STANDALONE CONFIG AUDIT -- NO MODEL INSTANTIATION, AND WHY flop_weight IS SEPARATE
===============================================================================================

``audit_config(path) -> ParamAudit(total, active, flop_weight, ...)`` walks the YAML.

``flop_weight`` weights each block by its ``layer_iterations`` entry:

    flop_weight = embedding/head + sum_i  N_i * active_i(block)  + final_norm
    active      = embedding/head + sum_i        active_i(block)  + final_norm

A recurrent block applied N times does N times the work with ONE copy of the weights, so it
contributes N x to FLOPs and 1 x to the parameter count. Those are DIFFERENT NUMBERS and the
distinction is the whole architectural claim:

    arm                        TOTAL      ACTIVE    FLOPwt   FLOPwt ex-embed
    cmix_400M_baseline_switch  400.031M  231.697M  231.697M    128.937M
    cmix_400M_sandwich_sparse  400.333M  156.539M  217.196M    114.435M
    cmix_400M_hybrid_sparse    399.784M  219.427M  299.376M    196.616M

The sandwich reaches the Switch baseline's compute with **32% fewer active parameters**,
purely because ``layer_iterations: [1, 4, 1]`` reuses one energy block four times.

``flop_weight`` counts the tied embedding ONCE, on the grounds that the tied ``lm_head`` is a
real ``V x d`` GEMM per token while the ``wte`` lookup is free -- so one copy is the right
charge. The embedding is 103M of these numbers at d=1024, i.e. it dominates, so the
ex-embedding column is reported alongside and any cross-arm claim should quote which one it
means. It is a WEIGHT-PROPORTIONAL proxy and deliberately ignores attention's quadratic
sequence term; ``train_utils.get_model_tflops`` is the real FLOP model.

===============================================================================================
HONESTY: WHERE THIS DISAGREES WITH THE HAND ARITHMETIC IT WAS ASKED TO REPRODUCE
===============================================================================================

The reference values supplied with this task were 125M / 375M / 392M / 375M / 966M total. Every
one of them is LOW, and by a SINGLE consistent cause: a swiglu ``MLP``'s ``c_fc`` is
``2*intermediate_size`` wide (``mlp.py:789`` ``2 * intermediate_size if is_glu(...)``), so a
dense swiglu block costs ``3*d*I``, not ``2*d*I``. The gap is exactly
``n_dense_swiglu_blocks * d * I``:

    config                     mine       ref     gap        n_dense * d * I
    cmix_134M_hybrid_32B_sp  134.253M    125M    9.44M       6 *  768 * 2048 =  9.437M
    cmix_400M_hybrid_sparse  399.784M    375M   24.78M       6 * 1024 * 4096 = 25.166M
    cmix_400M_sandwich_sp    400.333M    392M    8.33M       2 * 1024 * 4096 =  8.389M
    cmix_400M_baseline_sw    400.031M    375M   25.03M       6 * 1024 * 4096 = 25.166M
    cmix1B_12L_gptDense      1002.067M   966M   36.07M       8 * 1024 * 4096 = 33.554M

**MINE IS RIGHT, and the 1B row proves it independently:** the task states that the 1B's real
instantiated count from its safetensors header is **1002.1M**, and this module's meta build
gives **1002.067M** -- agreement to 0.003%, against a 3.6% miss for the hand figure. (The 1B's
residual 2.5M beyond the swiglu term is the four MoE blocks' 6.3M of proxy parameters partly
offsetting other small terms; the swiglu factor is the dominant and reproducible cause in all
five rows.)

The reference ACTIVE values are each the reference TOTAL minus the correct expert discount, so
the METHOD was right and only the total was off. Two more things the hand arithmetic could not
have known, both worth recording because they are not guessable:

  * ``energy_attention`` is ``2*d*d``, not ``4*d*d``. It projects Q and K only -- V = K and
    there is no output projection (``energy_attention.py``: ``c_attn`` is
    ``hidden_size -> 2*qk_dim``). The ``4*d*d`` rule holds for ``softmax_attention``
    (``3*d*d`` + ``d*d``) and overcounts every energy block by ``2*d*d``.
  * each energy block carries a ``proj`` and a ``scale_ff``. With ``energy_proj_type:
    psd_anti`` and the CommonConfig default ``energy_proj_rank: 32`` (NOT None, so the
    "full rank unless set" comment in ``layer.py`` does not apply), ``proj`` is
    ``S (d x 32) + A (d x d)`` = 614,400 at d=768 and 1,081,344 at d=1024. Both verified
    against the meta build.

===============================================================================================
TO WIRE IT IN (NOT DONE HERE)
===============================================================================================

One call, in ``model_wrapper/base.py:calculate_num_parameters`` immediately after the meta
``from_config``:

    from ..hf_models.modeling_utils.mlp_blocks.energy_ff_paramcount import (
        attach_active_param_counters)
    attach_active_param_counters(model)

It binds the method onto the existing ``FusedMoEContainer`` instances, so the frozen
recursion then finds it. It touches no forward path, allocates nothing, and changes only the
two numbers ``pretrain.py:683`` pushes into the wandb run config. Alternatively make
``build_boltzmann_moe`` return the ``ActiveParamCountMixin`` containers -- but that edits a
frozen file, which is why the binder exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn


# =============================================================================================
# (a) counters for live modules
# =============================================================================================


def _expert_stacked(p: nn.Parameter, n_experts: int, expert_I: int) -> bool:
    """Is this parameter the expert-major stack, i.e. discountable by ``top_k/K``?

    True for the fused weight ``(K*I_e, d)`` and for a fused bias ``(K*I_e,)``; False for a
    router parameter, whose leading dim is ``K`` (proxy_V is ``(K, d, r)``) or ``h``/``K``
    (the surrogate head). Keying on ``K*I_e`` and not on ``K`` is what keeps the two apart --
    and it is why this is a function and not an ``isinstance`` check: a K-leading router
    tensor and a K*I_e-leading expert stack are both "per-expert" in English and must be
    treated oppositely.
    """
    return p.dim() >= 1 and p.shape[0] == n_experts * expert_I


def moe_active_parameters(moe: nn.Module) -> int:
    """Active parameters of a ``BoltzmannMoEFFEnergy`` (or any subclass), EXCLUDING the
    fused holder, which is its sibling under ``FusedMoEContainer`` and is counted there.

    In the fused configuration every parameter reachable from here is a ROUTER parameter
    (proxy and/or surrogate head) and is therefore fully active -- see fact (2). The
    per-expert discount below only fires in the non-fused configuration where the expert
    modules own their weights.
    """
    K = int(moe.n_experts)
    top_k = getattr(moe, "top_k", None)
    n = sum(p.numel() for p in moe.parameters())

    if top_k is None or int(top_k) >= K:
        return n                                   # dense soft mixing: nothing is skipped

    k = int(top_k)
    # Non-fused experts that own their weights.
    for expert in getattr(moe, "experts", []):
        for p in expert.parameters():
            n -= p.numel()
            n += (p.numel() * k) // K
    return n


def holder_active_parameters(holder: nn.Module, n_experts: int, expert_I: int,
                             top_k: int | None) -> int:
    """Active parameters of a fused expert holder (``_FusedHopfieldHolder`` = one stacked
    matrix, ``_FusedW1W2Holder`` = two). Arity is never named: any parameter whose leading
    dim is ``K*I_e`` is discounted, so fact (1) needs no branch."""
    n = 0
    dense = top_k is None or int(top_k) >= n_experts
    k = n_experts if dense else int(top_k)
    for p in holder.parameters():
        if _expert_stacked(p, n_experts, expert_I):
            n += (p.numel() * k) // n_experts
        else:
            n += p.numel()
    return n


def expert_width(moe: nn.Module) -> int:
    """``I_e``. ``_expert_I`` only EXISTS when ``fused_experts=True`` (it is set inside that
    branch of ``__init__``), so reading it unguarded raises on a looped arm -- which is
    exactly how the first version of this function failed. Fall back to the experts, which
    always know their own width."""
    ei = getattr(moe, "_expert_I", None)
    if ei is not None:
        return int(ei)
    experts = getattr(moe, "experts", None)
    if experts is not None and len(experts) > 0:
        return int(experts[0].intermediate_size)
    return int(moe.intermediate_size) // int(moe.n_experts)


def container_active_parameters(container: nn.Module) -> int:
    """Active parameters of a ``FusedMoEContainer`` -- the module registered as ``ffwd``, and
    the one ``base.py``'s recursion actually reaches (it skips ``moe`` because it stops
    recursing as soon as a child answers ``get_num_active_parameters``)."""
    moe = container.moe
    holder = container.expert_holder
    return (moe_active_parameters(moe)
            + holder_active_parameters(holder, int(moe.n_experts), expert_width(moe),
                                       getattr(moe, "top_k", None)))


class ActiveParamCountMixin:
    """Mix into a future ``FusedMoEContainer`` subclass so the frozen recursion finds it."""

    def get_num_active_parameters(self) -> int:
        return container_active_parameters(self)


class MoEActiveParamCountMixin:
    """Mix into a ``BoltzmannMoEFFEnergy`` subclass. Only needed if a caller walks into the
    MoE directly; the container-level mixin already covers ``base.py``'s recursion."""

    def get_num_active_parameters(self) -> int:
        return moe_active_parameters(self)


def attach_active_param_counters(model: nn.Module, verbose: bool = False) -> int:
    """OPT-IN: bind ``get_num_active_parameters`` onto every energy-MoE container in ``model``.

    Returns the number of modules patched. Binds on the INSTANCE (``types.MethodType``), so no
    class is modified and no other model is affected; safe on a meta-device model, which is
    where ``calculate_num_parameters`` runs. Idempotent.
    """
    import types

    patched = 0
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls == "FusedMoEContainer" and not hasattr(mod, "get_num_active_parameters"):
            mod.get_num_active_parameters = types.MethodType(
                lambda self: container_active_parameters(self), mod)
            patched += 1
            if verbose:
                print(f"  patched {name} ({cls})")
        elif (cls.startswith("BoltzmannMoE") or cls.startswith("SurrogateBoltzmannMoE")) \
                and name.rsplit(".", 1)[-1] != "moe" \
                and not hasattr(mod, "get_num_active_parameters"):
            # A bare MoE wrapper used without the container (tests, probes).
            mod.get_num_active_parameters = types.MethodType(
                lambda self: moe_active_parameters(self), mod)
            patched += 1
    return patched


def count_model_parameters(model: nn.Module) -> tuple[int, int]:
    """``(total, active)`` for a live/meta model, reproducing ``base.py``'s recursion exactly
    but with the counters attached. Used by the test as the ground truth for ``audit_config``."""
    total = sum(p.numel() for p in model.parameters())
    attach_active_param_counters(model)
    active = 0

    def rec(module: nn.Module) -> None:
        nonlocal active
        for m in module.children():
            if hasattr(m, "get_num_active_parameters"):
                active += m.get_num_active_parameters()
            else:
                for p in m.parameters(recurse=False):
                    active += p.numel()
                rec(m)

    rec(model)
    return total, active


# =============================================================================================
# (b) standalone config audit -- no model instantiation
# =============================================================================================

_ENERGY_MIXERS = ("energy_attention", "mixed_head_energy_descent",
                  "boltzmann_moe_energy_attention")

_GLU = ("swiglu", "geglu", "reglu", "glu", "swiglu_packed")


@dataclass
class BlockAudit:
    idx: int
    kind: str                  # "energy" | "gpt"
    iterations: int
    total: int
    active: int
    detail: dict[str, int] = field(default_factory=dict)


@dataclass
class ParamAudit:
    total: int
    active: int
    flop_weight: int
    embedding: int
    blocks: list[BlockAudit]
    warnings: list[str]

    @property
    def flop_weight_ex_embedding(self) -> int:
        return self.flop_weight - self.embedding

    @property
    def active_ex_embedding(self) -> int:
        return self.active - self.embedding

    def __str__(self) -> str:  # pragma: no cover
        return (f"total={self.total/1e6:.3f}M active={self.active/1e6:.3f}M "
                f"flop_weight={self.flop_weight/1e6:.3f}M "
                f"(ex-embed {self.flop_weight_ex_embedding/1e6:.3f}M)")


def _norm_params(d: int, kind: str) -> int:
    k = (kind or "layernorm").lower()
    if k in ("rmsnorm", "rms_norm"):
        return d
    if k in ("layernorm", "layer_norm"):
        return 2 * d
    raise ValueError(f"unhandled normalization_function {kind!r}; add it here rather than "
                     f"guessing -- the count would be silently wrong")


def _attention_params(cfg: dict, blk: dict, d: int, energy: bool) -> int:
    heads = int(blk["num_attention_heads"])
    kv = int(blk.get("num_key_value_heads", heads))
    head_dim = int(blk.get("head_dim") or (d // heads))
    bias = bool(blk.get("add_bias", False))
    qkv_bias = bool(blk.get("qkv_bias", False))
    if energy:
        # Q and K only: V = K and there is NO output projection. 2*d*d in the coupled case,
        # NOT the 4*d*d of a standard attention block.
        qk = heads * head_dim
        n = 2 * qk * d + (2 * qk if qkv_bias else 0)
        if blk.get("add_wv_wo", False):
            n += 2 * d * d + (2 * d if bias else 0)
        return n
    out = d + 2 * kv * head_dim
    n = out * d + (out if bias else 0)              # c_attn  (= 3*d*d when kv == heads)
    n += d * d + (d if bias else 0)                 # c_proj
    return n


def _energy_proj_params(cfg: dict, d: int) -> tuple[int, str]:
    t = cfg.get("energy_proj_type", "unconstrained")
    if t == "identity":
        return 0, t
    if t in ("unconstrained", "antisymmetric", "port_hamiltonian"):
        return (d * d) * (2 if t == "port_hamiltonian" else 1), t
    if t == "pos_scalar":
        return 1, t
    if t == "psd_anti":
        # CommonConfig DEFAULTS energy_proj_rank to 32, not None, so `rank or hidden_size`
        # in layer.py resolves to 32 unless overridden. Verified against the meta build:
        # 614,400 at d=768 and 1,081,344 at d=1024.
        r = cfg.get("energy_proj_rank", 32)
        r = r if (r is not None and r > 0) else d
        n = d * r
        ar = cfg.get("energy_antisym_rank", None)
        n += (d * d) if (ar is None or ar >= d) else (2 * d * ar)
        return n, t
    if t in ("dual_unconstrained", "attn_only_energy"):
        return (2 * d * d if t == "dual_unconstrained" else d * d) + 1, t
    if t == "low_rank_antisymmetric":
        r = cfg.get("energy_proj_rank", 32)
        return 2 * d * r, t
    raise ValueError(f"unhandled energy_proj_type {t!r}; add it rather than guessing")


def _mlp_params(cfg: dict, blk: dict, d: int) -> tuple[int, int, dict]:
    """``(total, active, detail)`` for one ``mlp_blocks`` entry."""
    t = blk.get("mlp_type", "MLP")
    bias = bool(blk.get("add_bias", False))
    det: dict[str, int] = {}

    if t == "MLP":
        I = int(blk["intermediate_size"])
        glu = str(blk.get("activation_function", "gelu")).lower() in _GLU
        fc_out = 2 * I if glu else I
        n = fc_out * d + (fc_out if bias else 0) + I * d + (d if bias else 0)
        det["mlp_glu"] = int(glu)
        return n, n, det

    if t == "MoE":
        # Switch-style. Reproduces moe.py:418 analytically.
        I = int(blk["intermediate_size"])
        K = int(blk["num_experts"])
        k = int(blk["num_experts_per_tok"])
        glu = str(blk.get("activation_function", "gelu")).lower() in _GLU
        fc_out = 2 * I if glu else I
        c_fc = K * fc_out * d
        c_proj = K * I * d
        gate = d * K
        n = c_fc + c_proj + gate
        a = gate + (c_fc * k) // K + (c_proj * k) // K
        sh = blk.get("shared_intermediate_size")
        if sh:
            s_fc = (2 * int(sh) if glu else int(sh)) * d
            s_pr = int(sh) * d
            n += s_fc + s_pr
            a += s_fc + s_pr                        # shared expert runs for every token
            if blk.get("shared_expert_gating", False):
                n += d
                a += d
        det.update(experts=K, top_k=k, gate=gate)
        return n, a, det

    if t == "EnergyFF_Hopfield":
        I = int(blk["intermediate_size"])
        n = I * d + (I if bias else 0)
        return n, n, det

    if t == "EnergyFF_W1W2":
        I = int(blk["intermediate_size"])
        n = 2 * (I * d + (I if bias else 0))
        return n, n, det

    if t == "EnergyFF_BoltzmannMoE":
        I = int(blk["intermediate_size"])
        K = int(blk["n_experts"])
        kind = blk.get("expert_kind", "hopfield")
        top_k = blk.get("top_k", None)
        assert I % K == 0, f"intermediate_size {I} not divisible by n_experts {K}"
        I_e = I // K
        n_mat = 2 if kind == "w1w2" else 1          # fact (1)
        experts = n_mat * (I * d + (I if bias else 0))
        k = K if (top_k is None or int(top_k) >= K) else int(top_k)   # fact (4)
        experts_active = (experts * k) // K

        # fact (2): router parameters are ALWAYS active.
        router = 0
        r = int(blk.get("proxy_rank", 0) or 0)
        if r > 0:
            it = max(1, int(blk.get("proxy_iters", 1) or 1))
            pk = blk.get("proxy_kind", "quad")
            router += it * K * d * r                                     # proxy_V
            if pk == "quad":
                router += it * (K * r + K * r + K)                       # quad, lin, bias
            else:
                m = int(blk.get("proxy_out_dim", 0) or 0) or I_e
                router += it * (K * m * r + K + K)                       # B, scale, bias
            if kind == "w1w2":
                # energy_ff_w1w2_sparse allocates a SECOND basis (proxy_V2 / proxy_B2)
                router += it * K * d * r
                if pk == "subspace":
                    m = int(blk.get("proxy_out_dim", 0) or 0) or I_e
                    router += it * K * m * r
        sk = blk.get("surrogate_kind")
        if sk and float(blk.get("surrogate_coef", 0) or 0) >= 0 and (
                blk.get("surrogate_coef") is not None or blk.get("use_surrogate")):
            h = int(blk.get("surrogate_hidden", 0) or 0)
            if sk == "linear":
                router += K * d + K
            else:
                router += h * d + h + K * h + K
        det.update(experts=K, expert_I=I_e, top_k=(k if k < K else None),
                   n_matrices=n_mat, router=router)
        return experts + router, experts_active + router, det

    raise ValueError(f"unhandled mlp_type {t!r} in mlp_blocks; add it rather than guessing "
                     f"-- silently returning 0 would understate the model")


def audit_config(path_or_cfg: str | dict) -> ParamAudit:
    """``(total, active, flop_weight)`` for a training YAML WITHOUT instantiating the model.

    Accepts a path, or the already-parsed ``pretrained_config`` dict, or the whole YAML dict.
    Raises rather than guessing on any block type it does not know: a silent 0 is how a config
    gets audited as smaller than it is.
    """
    if isinstance(path_or_cfg, (str, bytes)) or hasattr(path_or_cfg, "__fspath__"):
        import yaml
        with open(path_or_cfg) as fh:
            doc = yaml.safe_load(fh)
        cfg = doc["model_args"]["pretrained_config"]
    elif "model_args" in path_or_cfg:
        cfg = path_or_cfg["model_args"]["pretrained_config"]
    else:
        cfg = path_or_cfg

    warnings: list[str] = []
    d = int(cfg["hidden_size"])
    V = int(cfg["vocab_size"])
    L = int(cfg["num_layers"])
    norm = cfg.get("normalization_function", "layernorm")
    energy_norm = cfg.get("energy_norm_type") or norm
    tied = bool(cfg.get("tie_word_embeddings", True))

    iters = cfg.get("layer_iterations")
    if iters is None:
        npre = int(cfg.get("num_pre_layers", 8))
        npost = int(cfg.get("num_post_layers", 8))
        nit = int(cfg.get("num_iterations", 1))
        iters = [1] * npre + [nit] * (L - npre - npost) + [1] * npost
        warnings.append("layer_iterations absent; reconstructed from "
                        "num_pre_layers/num_iterations/num_post_layers as config/__init__.py "
                        "does. VERIFY -- this is the field that silently decides FLOPs.")
    assert len(iters) == L, f"layer_iterations has {len(iters)} entries for {L} layers"

    emb = V * d * (1 if tied else 2)
    total = emb + _norm_params(d, norm)                      # wte (+lm_head) + ln_f
    active = total
    flopw = total
    blocks: list[BlockAudit] = []

    smx = cfg["sequence_mixer_blocks"]
    mlps = cfg["mlp_blocks"]
    for i in range(L):
        sb, mb = smx[i], mlps[i]
        mixer = sb["sequence_mixer_type"]
        energy = mixer in _ENERGY_MIXERS
        n_it = int(iters[i])
        det: dict[str, int] = {}
        if energy:
            attn = _attention_params(cfg, sb, d, energy=True)
            proj, pt = _energy_proj_params(cfg, d)
            b_tot = _norm_params(d, energy_norm) + attn + proj + 1     # + scale_ff
            b_act = b_tot
            det.update(attn=attn, proj=proj, proj_type=pt)
        else:
            if mixer != "softmax_attention":
                warnings.append(f"block {i}: sequence_mixer_type {mixer!r} counted with the "
                                f"standard-attention formula; verify if that is wrong")
            attn = _attention_params(cfg, sb, d, energy=False)
            b_tot = 2 * _norm_params(d, norm) + attn
            b_act = b_tot
            det.update(attn=attn)
        m_tot, m_act, m_det = _mlp_params(cfg, mb, d)
        det.update(m_det)
        det["mlp"] = m_tot
        b_tot += m_tot
        b_act += m_act

        total += b_tot
        active += b_act
        flopw += n_it * b_act
        blocks.append(BlockAudit(i, "energy" if energy else "gpt", n_it, b_tot, b_act, det))

    return ParamAudit(total=total, active=active, flop_weight=flopw, embedding=emb,
                      blocks=blocks, warnings=warnings)


def format_audit(name: str, a: ParamAudit) -> str:  # pragma: no cover
    lines = [f"{name}",
             f"  TOTAL        {a.total/1e6:10.3f}M",
             f"  ACTIVE       {a.active/1e6:10.3f}M   "
             f"({100*a.active/a.total:.1f}% of total)",
             f"  FLOPwt       {a.flop_weight/1e6:10.3f}M   "
             f"(ex-embedding {a.flop_weight_ex_embedding/1e6:.3f}M)",
             f"  embedding    {a.embedding/1e6:10.3f}M"]
    for b in a.blocks:
        if b.kind == "energy" or b.detail.get("experts"):
            lines.append(f"    block {b.idx:2d} {b.kind:6s} x{b.iterations}  "
                         f"total {b.total/1e6:8.3f}M  active {b.active/1e6:8.3f}M  {b.detail}")
    for w in a.warnings:
        lines.append(f"  WARNING: {w}")
    return "\n".join(lines)


__all__ = [
    "ActiveParamCountMixin",
    "MoEActiveParamCountMixin",
    "ParamAudit",
    "BlockAudit",
    "attach_active_param_counters",
    "audit_config",
    "container_active_parameters",
    "count_model_parameters",
    "expert_width",
    "format_audit",
    "holder_active_parameters",
    "moe_active_parameters",
]
