# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Any

from ...utils import BaseArgs


class _EnergyMLPArgs(BaseArgs):
    mlp_type: str = "Energy_MLP"
    intermediate_size: int
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "Energy_MLP"


class _HopfieldEnergyMLPArgs(BaseArgs):
    """Config for Hopfield_Energy_MLP: bounded-below FFN energy E_FF = ||gelu(Wh)||^2.

    Single weight W of shape [intermediate, hidden] — half the per-block FFN
    params of Energy_MLP (which has W1 + W2). To match the Energy_MLP param
    count, double the intermediate_size at config time (e.g. 4096 → 8192).
    """
    mlp_type: str = "Hopfield_Energy_MLP"
    intermediate_size: int
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "Hopfield_Energy_MLP"


class _BoltzmannMoEHopfieldEnergyMLPArgs(BaseArgs):
    """Config for BoltzmannMoE_Hopfield_Energy_MLP.

    K experts, each with the bounded-below single-W Hopfield form
    E_k = (1/expert_I) ||gelu(W_k h)||^2 ≥ 0.  Boltzmann routing
    w_k = softmax(-E_k); total energy E_total = -LSE_k(-E_k) is bounded
    both directions (∈ [-log K, min_k E_k]).

    Iso-parameter with Hopfield_Energy_MLP at the same intermediate_size:
    `intermediate_size` is the TOTAL across all experts; each expert gets
    `intermediate_size / n_experts` neurons.  e.g. intermediate_size=8192,
    n_experts=8 ⇒ 8 experts × 1024 neurons each (iso-param with the
    Hopfield-MEAN big run).
    """

    mlp_type: str = "BoltzmannMoE_Hopfield_Energy_MLP"
    intermediate_size: int   # total across all experts = n_experts * expert_I
    n_experts: int = 8
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "BoltzmannMoE_Hopfield_Energy_MLP"
        assert self.n_experts >= 2, "BoltzmannMoE_Hopfield requires at least 2 experts"
        assert self.intermediate_size % self.n_experts == 0, (
            f"intermediate_size ({self.intermediate_size}) must be divisible by "
            f"n_experts ({self.n_experts})"
        )


class _MLPArgs(BaseArgs):
    mlp_type: str = "MLP"
    intermediate_size: int
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MLP"


class _CompositionalEnergyMLPArgs(BaseArgs):
    mlp_type: str = "Compositional_Energy_MLP"
    intermediate_size: int
    num_paths: int = 4
    path_activations: list[str] | None = None
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "Compositional_Energy_MLP"
        if self.path_activations is not None and len(self.path_activations) == 0:
            self.path_activations = None
        if self.path_activations is not None:
            assert len(self.path_activations) == self.num_paths, (
                f"path_activations length ({len(self.path_activations)}) must match num_paths ({self.num_paths})"
            )
        assert self.intermediate_size % self.num_paths == 0, (
            f"intermediate_size ({self.intermediate_size}) must be divisible by num_paths ({self.num_paths})"
        )


class _MixedEnergyMLPArgs(BaseArgs):
    """Config for Mixed_Energy_MLP: half Energy_MLP + half standard MLP.

    Iso-param sizing: energy_intermediate_size + standard_intermediate_size = 1.5 * base_intermediate_size
    gives the same param count as a SwiGLU MLP with base_intermediate_size.
    Example: base=1536 → energy=1152, standard=1152 (GELU).
    """
    mlp_type: str = "Mixed_Energy_MLP"
    intermediate_size: int = 0  # unused; required by base class machinery
    energy_intermediate_size: int = 1152
    standard_intermediate_size: int = 1152
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "Mixed_Energy_MLP"
        if self.intermediate_size == 0:
            self.intermediate_size = self.energy_intermediate_size + self.standard_intermediate_size


class _MoEArgs(_MLPArgs):
    mlp_type: str = "MoE"
    shared_intermediate_size: int | None = None
    num_experts: int = 8
    use_interleaved_weights: bool = False
    num_experts_per_tok: int = 2
    shared_expert_gating: bool = False
    normalized_topk: bool = True

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE"


class _BoltzmannMoEEnergyMLPArgs(BaseArgs):
    """Config for BoltzmannMoE_Energy_MLP.

    Iso-parameter with Energy_MLP: total FLOPs and params equal one Energy_MLP with the
    same intermediate_size.  Each expert receives intermediate_size // n_experts neurons.

    For ~400M params with d=768, 12 blocks: intermediate_size=16384, n_experts=16
    gives 16 experts × 1024 neurons each.
    """

    mlp_type: str = "BoltzmannMoE_Energy_MLP"
    intermediate_size: int  # total across all experts = n_experts * per_expert_I
    n_experts: int = 8
    temperature: float = 1.0
    repulsion_coef: float = 0.0      # 0 = disabled; try 0.01 for stochastic repulsion
    n_repulsion_pairs: int = 4
    # DEFAULT "signed" FOR THIS LEGACY CLASS, deliberately different from the composable
    # class's "abs". Every arm built on BoltzmannMoE_Energy_MLP -- including the published h1
    # models -- was TRAINED with the old hardcoded cos.mean(), which is exactly "signed". The
    # field did not exist here, so the module fell back to getattr(..., "squared") and those
    # configs silently changed behaviour with no way to pin them back. Defaulting to "signed"
    # makes legacy arms reproduce by construction; set "abs" explicitly for new work (it is
    # worth +0.69pp / -1.83 ppl, and is what the composable class now defaults to).
    repulsion_form: str = "signed"
    top_k: int | None = None    # None = soft (all experts active); int = sparse top-k Boltzmann routing
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0.0
    add_bias: bool = False
    # Gradient-of-φ approximation used when computing the second-half gradient
    # term2 = W1ᵀ (φ'(W1 h) ⊙ W2 h).
    #   "sigmoid"    : phi' = sigmoid(√(2/π) · W1 h) × 0.5      (LEGACY default; not the
    #                  derivative of F.gelu and uniformly half the true GELU' magnitude;
    #                  partly absorbed by W2's learned scale).
    #   "erf_exact"  : phi  = F.gelu(W1 h) (unchanged), phi' = analytic d/dx F.gelu(x)
    #                  = 0.5·(1 + erf(x/√2)) + x·exp(-x²/2)/√(2π). Cleanest A/B
    #                  vs sigmoid: only the φ' magnitude/shape changes, φ identical.
    #   "tanh_exact" : matched φ = 0.5 W1 h (1 + tanh(c · W1 h)) and exact φ' for that φ
    #                  (c = √(2/π)). Self-consistent ∂E/∂h. Tested at h1 scale;
    #                  lost −2.2pp avg vs sigmoid (negative result).
    # Default = "sigmoid" so existing checkpoints (V1, h1_*, B-series, 580M) load and
    # produce identical outputs.
    gelu_grad_method: str = "sigmoid"

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "BoltzmannMoE_Energy_MLP"
        assert self.n_experts >= 2, "BoltzmannMoE requires at least 2 experts"
        assert self.intermediate_size % self.n_experts == 0, (
            f"intermediate_size ({self.intermediate_size}) must be divisible by "
            f"n_experts ({self.n_experts})"
        )
        assert self.temperature > 0, "temperature must be positive"
        assert self.gelu_grad_method in ("sigmoid", "tanh_exact", "erf_exact"), (
            f"gelu_grad_method must be one of 'sigmoid' / 'tanh_exact' / 'erf_exact', "
            f"got {self.gelu_grad_method}"
        )


class _TopKEnergyMoEMLPArgs(BaseArgs):
    """Config for TopK_Energy_MoE_MLP.

    Each of n_experts experts gets the full intermediate_size (not divided by n_experts).
    top_k experts are selected per token by a learned linear router.
    load_balance_coef: auxiliary load-balancing loss weight (Switch §2.1). Use 0.01.
    """

    mlp_type: str = "TopK_Energy_MoE_MLP"
    intermediate_size: int   # per-expert (full size, NOT divided by n_experts)
    n_experts: int = 4
    top_k: int = 2
    load_balance_coef: float = 0.01   # prevents routing collapse; 0.01 follows Switch
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0.0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "TopK_Energy_MoE_MLP"
        assert 1 <= self.top_k <= self.n_experts, (
            f"top_k ({self.top_k}) must be in [1, n_experts ({self.n_experts})]"
        )


class _EnergyFFW1W2Args(BaseArgs):
    """Config for the new composable W1W2 FF energy class (``EnergyFF_W1W2``).

    Drop-in replacement for ``Energy_MLP`` — same param count, same math,
    different class. ``gelu_grad_method`` selects sigmoid (legacy) /
    tanh_exact / erf_exact.
    """
    mlp_type: str = "EnergyFF_W1W2"
    intermediate_size: int
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False
    gelu_grad_method: str = "sigmoid"

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "EnergyFF_W1W2"
        assert self.gelu_grad_method in ("sigmoid", "tanh_exact", "erf_exact")


class _EnergyFFHopfieldArgs(BaseArgs):
    """Config for the new composable Hopfield FF energy class (``EnergyFF_Hopfield``).

    Drop-in replacement for ``Hopfield_Energy_MLP``. ``E = (1/d_int)||gelu(Wh)||²``.
    """
    mlp_type: str = "EnergyFF_Hopfield"
    intermediate_size: int
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False
    gelu_grad_method: str = "sigmoid"

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "EnergyFF_Hopfield"
        assert self.gelu_grad_method in ("sigmoid", "tanh_exact", "erf_exact")


class _EnergyFFBoltzmannMoEArgs(BaseArgs):
    """Config for the new composable Boltzmann-MoE FF energy (``EnergyFF_BoltzmannMoE``).

    Routes K experts of either ``w1w2`` or ``hopfield`` kind via softmax(±E_k/τ)
    + optional stochastic repulsion. This is the variant that **adds the
    repulsion + τ + n_repulsion_pairs** to the Hopfield-MoE form (which the
    legacy ``BoltzmannMoE_Hopfield_Energy_MLP`` was missing).

    Iso-parameter with the corresponding non-MoE expert kind at the same
    ``intermediate_size``: ``n_experts × (intermediate_size / n_experts)``
    total neurons.
    """
    mlp_type: str = "EnergyFF_BoltzmannMoE"
    intermediate_size: int
    n_experts: int = 8
    expert_kind: str = "hopfield"     # "w1w2" or "hopfield"
    temperature: float = 1.0
    repulsion_coef: float = 0.0
    n_repulsion_pairs: int = 4
    # "squared"|"abs"|"hinge" are minimised at orthogonality. "signed" is the
    # pre-2026-09-12 legacy form, minimised at cos=-1, which rewards
    # anti-alignment and collapses the FF branch under near-uniform routing.
    # DEFAULT IS "abs" (2026-09-14). Three facts forced this:
    #  * the pre-2026-09-12 code hardcoded cos.mean(), i.e. "signed", which is MINIMISED at
    #    cos = -1 and therefore rewarded anti-aligned experts -- a mis-specification;
    #  * "abs" is the only form we have actually trained (13 arms) and is worth +0.69pp /
    #    -1.83 ppl over "signed" (45.89 vs 45.20);
    #  * "squared" was never trained by anyone, and at our measured expert cosines
    #    (|cos| <= 0.067) it is ~15x weaker than "abs", so defaulting to it silently
    #    near-disables repulsion for any config that omits this key.
    # Legacy configs under configs/boltzmann_moe/ now pin "signed" explicitly so they
    # reproduce what they were trained with.
    repulsion_form: str = "abs"
    # "none" keeps softmax(+-E/tau) as trained. The Hopfield MEAN form leaves
    # E ~ 1e-2 against tau=1 and routes uniformly; "zscore" (scale-free) or
    # "sqrt_width" (matches the w1w2 line's 1/sqrt(expert_I)) fix that.
    routing_norm: str = "none"
    # Prefactor on the Hopfield descent gradient. "mean" = 4/I_e = the
    # pre-2026-09-12 behaviour; SET THIS BACK TO "mean" TO REVERT if a run
    # diverges. "inv_sqrt" = 1/sqrt(I_e) (8x), "sqrt_consistent" = 4/sqrt(I_e)
    # (32x). Both keep >=32x margin below the sum form that NaN'd (run 1714840).
    # See _hopfield_grad_prefactor in energy_ff.py for the full rationale.
    hopfield_grad_scale: str = "mean"
    top_k: int | None = None
    # Match the baselines' sum(p)=1 on the top-k weights (TopK_Energy_MoE_MLP softmaxes over
    # the top-k logits; the Switch class uses normalized_topk). Our masked form leaves
    # sum(p) ~ 0.45 at K=16,k=2. Default false = unchanged behaviour.
    renormalize_topk: bool = False

    # ---------------- acceleration knobs (boltz-accel, 2026-09-15) ----------
    # All default to the pre-existing behaviour, so untouched configs and
    # existing checkpoints are bit-identical.
    #
    # repulsion_interval: fire repulsion on ~1 call in N (stochastic). With
    #   repulsion_scale_comp the coefficient is multiplied by N on firing calls,
    #   so the time-averaged repulsion pressure is unchanged. 1 = every call.
    repulsion_interval: int = 1
    repulsion_scale_comp: bool = True
    # fused_experts: EXACT single-GEMM expert path (hopfield only). Collapses the
    #   2*K per-expert slice-GEMMs into 2 GEMMs against the shared fused weight.
    #   Identical arithmetic -- the loss curve must match. This is the real
    #   speedup: the profiler measured 79.5k kernels/step and GPU-busy 1.71s
    #   against 6.75s wall, i.e. the cost is launch overhead, not FLOPs.
    fused_experts: bool = False
    # proxy_rank: rank r of the learnable cheap router (0 = disabled). The proxy
    #   is trained online by KL against the exact routing distribution and its
    #   top-k agreement is logged as `proxy_topk_agree`. It does NOT affect the
    #   forward pass unless proxy_route=True, so enabling it cannot degrade
    #   training -- it only costs K*d*r MACs/token (~3% at r=8).
    proxy_rank: int = 0
    proxy_loss_coef: float = 0.0
    proxy_route: bool = False
    # cos_probe_interval: measure mean|cos| between expert outputs under no_grad on
    #   1 call in N, INDEPENDENTLY of the repulsion loss, and log it as
    #   `expert_cos_abs_mean`. 0 = off. Needed because the repulsion aux loss is
    #   coef*mean|cos| and is therefore unreadable at coef=0 -- which is the control
    #   that says how much of the alignment plateau repulsion actually buys.
    cos_probe_interval: int = 0
    cos_probe_pairs: int = 8
    # repulsion_space: "output" (default, as trained -- cosine between per-token
    #   expert outputs, cost scales with N) or "weight" (cosine between expert
    #   weight blocks: no token dimension, so 3.5x cheaper at N=4096 and 6.4x at
    #   N=8192, and sparse-kernel compatible). NOTE repulsion_coef does NOT
    #   transfer between the two spaces -- weight cosines are ~5-25x smaller than
    #   output cosines -- so re-sweep it when switching.
    repulsion_space: str = "output"
    # e_sign_override: None (default) keeps the kind-based routing sign every
    #   existing checkpoint trained with. "pos" makes softmax favour the LARGEST
    #   stored energy, "neg" the smallest. For hopfield experts the stored energy
    #   GROWS with overlap, so "pos" routes to the BEST-matching experts and the
    #   default "neg" routes to the worst -- see ROUTING_SIGN_BUG_20260915.md.
    e_sign_override: str | None = None
    # sinkhorn_iters: 0 (default) = off. >0 solves the EXACT dual variables
    #   (chemical potentials) that equalise expert load, by log-domain Sinkhorn:
    #       mu <- mu + log(load(mu) * K)
    #   This is the same constraint `balance_rate` targets, but solved rather than
    #   controlled -- no +-1.0 clamp to saturate against (both balance_rate arms in
    #   the corrected-sign sweep pinned at the clamp), no gain to tune, and exact
    #   rather than lagged. Solved under no_grad, so no gradient pathway and no
    #   auxiliary loss. TRAIN-ONLY (load is a batch property), and the load is the
    #   LOCAL per-rank batch -- see ROUTING_SIGN_BUG_20260915.md.
    #   Mutually exclusive with balance_rate: both solve the same constraint.
    sinkhorn_iters: int = 0
    # Keep a running estimate of the Sinkhorn dual and USE IT AT EVAL. Off by default:
    # it adds a persistent buffer, and the checkpoint loader is strict, so enabling it on a
    # run that already has checkpoints breaks resume. See energy_ff.py for the 5.33pp
    # measurement that motivates it.
    sinkhorn_persist_mu: bool = False
    # repulsion_tensor_idx: draw the repulsion pair indices with torch RNG into a TENSOR
    #   instead of Python `random` into lists. dynamo cannot trace Python random and
    #   specialises on the list values, so repulsion currently costs 2-3 graph breaks and
    #   ~17 recompiles per 8 calls -- in the EXISTING looped path too, and it is the
    #   leading suspect for the fused_experts multi-node hang. Tensor indices are data,
    #   not graph constants. Also makes activation-checkpoint recompute draw the SAME
    #   pairs as the forward (torch RNG is restored, Python's is not). Default OFF
    #   because it changes which pairs are drawn.
    repulsion_tensor_idx: bool = False
    # Accumulate routing load in-graph so it is logged even under torch_compile, where the
    # older _log_metrics path is traced away (which is why routing collapse went unseen).
    track_load: bool = True
    # >0 enables aux-loss-FREE balancing (DeepSeek-V3 style): a per-expert additive logit
    # bias nudged under no_grad toward under-loaded experts. Not a loss, no gate params.
    # DEFAULT 0 = OFF: enabling it changes the routing of every existing checkpoint.
    balance_rate: float = 0.0
    gelu_grad_method: str = "sigmoid"
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "EnergyFF_BoltzmannMoE"
        assert self.expert_kind in ("w1w2", "hopfield")
        assert self.n_experts >= 2
        assert self.intermediate_size % self.n_experts == 0, (
            f"intermediate_size ({self.intermediate_size}) must be divisible by "
            f"n_experts ({self.n_experts})"
        )
        assert self.temperature > 0
        assert self.repulsion_form in ("squared", "abs", "hinge", "signed")
        assert self.routing_norm in ("none", "zscore", "sqrt_width")
        assert self.hopfield_grad_scale in ("mean", "inv_sqrt", "sqrt_consistent")
        assert self.gelu_grad_method in ("sigmoid", "tanh_exact", "erf_exact")


class _SurrogateBoltzmannMoEMLPArgs(BaseArgs):
    """Config for SurrogateBoltzmannMoE_Energy_MLP.

    Iso-parameter with Energy_MLP (intermediate_size = n_experts * per_expert_I).
    Adds a linear surrogate router trained to mimic Boltzmann routing via KL distillation.
    """

    mlp_type: str = "SurrogateBoltzmannMoE_Energy_MLP"
    intermediate_size: int   # total = n_experts * per_expert_I (iso-param)
    n_experts: int = 16
    temperature: float = 1.0
    repulsion_coef: float = 0.0
    n_repulsion_pairs: int = 4
    surrogate_coef: float = 1.0   # weight of KL(surrogate || boltzmann) distillation loss
    use_surrogate: bool = True    # use linear router at eval time (cheap inference)
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0.0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "SurrogateBoltzmannMoE_Energy_MLP"
        assert self.n_experts >= 2
        assert self.intermediate_size % self.n_experts == 0, (
            f"intermediate_size ({self.intermediate_size}) must be divisible by n_experts ({self.n_experts})"
        )
        assert self.temperature > 0
