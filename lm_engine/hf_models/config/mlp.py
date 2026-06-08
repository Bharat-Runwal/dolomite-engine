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


class _MoEEnergyArgs(_EnergyMLPArgs):
    mlp_type: str = "MoE_Energy"
    shared_intermediate_size: int | None = None
    num_experts: int = 8
    num_experts_per_tok: int = 2
    shared_expert_gating: bool = False
    normalized_topk: bool = True
    stop_gradient_routing: bool = False  # detach router weights to make MoE energy-compatible

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE_Energy"


class _MoEEnergyModuleArgs(_EnergyMLPArgs):
    mlp_type: str = "MoE_Energy_Module"
    shared_intermediate_size: int | None = None
    num_experts: int = 8
    num_experts_per_tok: int = 2
    shared_expert_gating: bool = False
    normalized_topk: bool = True
    energy_routing: bool = False        # True: route by expert energies (exact free-energy interpretation)
    energy_routing_tau: float = 1.0     # temperature for free-energy routing softmax

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE_Energy_Module"


class _MoEEnergyF5Args(_EnergyMLPArgs):
    mlp_type: str = "MoE_Energy_F5"
    shared_intermediate_size: int | None = None
    num_experts: int = 8
    num_experts_per_tok: int = 2
    shared_expert_gating: bool = False
    normalized_topk: bool = True
    boltzmann_temperature: float = 1.0      # initial Boltzmann temperature
    learnable_temperature: bool = True       # make temperature a learnable parameter
    distillation_weight: float = 0.01        # weight for KL distillation loss
    use_boltzmann_at_inference: bool = False  # override to use Boltzmann routing at inference
    expert_repulsion_lambda: float = 0.0     # cosine diversity penalty on expert W1 weights
    # Stabilize aux loss path: detach expert_energies in router_logits_for_aux
    # AND zero out z_loss inside switch loss. Eliminates the squared-logsumexp
    # gradient-explosion vector that drives early-training spikes. Default False
    # for back-compat with prior runs.
    aux_loss_stable: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE_Energy_F5"


class _MoEEnergyF6Args(_EnergyMLPArgs):
    mlp_type: str = "MoE_Energy_F6"
    shared_intermediate_size: int | None = None
    num_experts: int = 8
    num_experts_per_tok: int = 2
    shared_expert_gating: bool = False
    normalized_topk: bool = True
    boltzmann_temperature: float = 1.0      # initial Boltzmann temperature
    learnable_temperature: bool = True       # make temperature a learnable parameter
    distillation_weight: float = 0.01        # weight for KL distillation loss
    use_boltzmann_at_inference: bool = False  # override to use Boltzmann routing at inference

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE_Energy_F6"


class _GaussianBoltzmannMoEArgs(BaseArgs):
    mlp_type: str = "GaussianBoltzmannMoE"
    intermediate_size: int
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False
    num_experts: int = 32
    num_experts_per_tok: int = 2
    normalized_topk: bool = True
    use_det_normalization: bool = True       # include ½ log det(W^T W) in routing logits
    use_mixing_coefficients: bool = True     # learnable GMM mixing weights π_e
    diversity_lambda: float = 0.01           # cosine diversity penalty on W matrices
    entropy_bonus_gamma: float = 0.0         # per-token routing entropy bonus
    mixing_entropy_gamma: float = 0.0        # entropy bonus on π_e mixing coefficients
    centroid_repulsion_lambda: float = 0.0   # pairwise centroid repulsion penalty
    bias_based_balancing: bool = False        # DeepSeek-V3 aux-loss-free balancing via per-expert bias
    bias_update_alpha: float = 0.001          # step size for bias adjustment
    # KL distillation to linear router (F5-style inference efficiency)
    kl_distillation: bool = False             # enable KL distillation to a cheap linear gate
    distillation_weight: float = 0.01         # weight for KL divergence loss
    use_gmm_at_inference: bool = False        # override: use full GMM routing at inference too

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "GaussianBoltzmannMoE"

class _BoltzmannMoEEnergyMLPArgs(BaseArgs):
    """Iso-parameter Boltzmann MoE Energy FFN.

    No linear router, no Switch aux/z-loss, no KL distillation. Routing comes
    directly from per-expert energies (softmax over E_i = phi(W1_i h)^T W2_i h),
    which sidesteps the squared-logsumexp gradient explosion that drove the
    early-training spikes in MoE_Energy_F5.

    intermediate_size is the TOTAL across all experts; each expert gets
    intermediate_size // n_experts hidden units.
    """

    mlp_type: str = "BoltzmannMoE_Energy_MLP"
    intermediate_size: int  # total across all experts = n_experts * per_expert_I
    n_experts: int = 8
    temperature: float = 1.0
    repulsion_coef: float = 0.0      # 0 = disabled; try 0.01 for stochastic repulsion
    n_repulsion_pairs: int = 4
    # None or 0 = soft (all experts active); int>0 = sparse top-k Boltzmann routing
    top_k: int | None = None
    gelu_grad_method: str = "sigmoid"
    activation_function: str = "gelu"
    dropout: float = 0.0
    add_bias: bool = False
    # Backward-compat: checkpoints trained with the EARLIER BoltzmannMoE port carry
    # `normalized_topk` in their saved config. Nima's vendored class uses no-renorm
    # truncation and does NOT consume this field, but we must ACCEPT it so those old
    # checkpoints still load/eval. Accepted-but-ignored.
    normalized_topk: bool = True

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "BoltzmannMoE_Energy_MLP"
        assert self.n_experts >= 2, "BoltzmannMoE requires at least 2 experts"
        assert self.intermediate_size % self.n_experts == 0, (
            f"intermediate_size ({self.intermediate_size}) must be divisible by "
            f"n_experts ({self.n_experts})"
        )
        assert self.temperature > 0, "temperature must be positive"
        assert self.top_k is None or 0 <= self.top_k <= self.n_experts, (
            f"top_k ({self.top_k}) must be None or in [0, n_experts={self.n_experts}]"
        )
        assert self.gelu_grad_method in ("sigmoid", "tanh_exact", "erf_exact"), (
            f"gelu_grad_method must be one of 'sigmoid' / 'tanh_exact' / 'erf_exact', "
            f"got {self.gelu_grad_method}"
        )


class _TopKEnergyMoEMLPArgs(BaseArgs):
    """Top-K Energy MoE — Nima's TopK_Energy_MoE_MLP.

    Each expert is a FULL-SIZE Energy_MLP (intermediate_size NOT divided).
    Linear router selects top_k; Switch-style load_balance_coef is the only
    aux loss (no z-loss, no KL distillation, no repulsion).
    """

    mlp_type: str = "TopK_Energy_MoE_MLP"
    intermediate_size: int                # per-expert (NOT divided by n_experts)
    n_experts: int = 4
    top_k: int = 2
    load_balance_coef: float = 0.01
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0.0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "TopK_Energy_MoE_MLP"
        assert self.n_experts >= 2, "TopK_Energy_MoE_MLP requires at least 2 experts"
        assert 1 <= self.top_k <= self.n_experts, (
            f"top_k ({self.top_k}) must be in [1, n_experts={self.n_experts}]"
        )


class _BoltzRouterTopKEnergyMoEMLPArgs(BaseArgs):
    """fh1 with a Boltzmann router instead of a linear top-k router.

    Same per-expert energy-gradient form as TopK_Energy_MoE_MLP. Routing is
    softmax over -E_e(x) / T with learnable T. Switch load-balance only;
    no z-loss, no KL distillation, no repulsion.
    """

    mlp_type: str = "BoltzRouter_TopK_Energy_MoE_MLP"
    intermediate_size: int                # per-expert (NOT divided by n_experts)
    n_experts: int = 8
    top_k: int = 2
    load_balance_coef: float = 0.01
    boltzmann_temperature: float = 1.0
    learnable_temperature: bool = True
    energy_scale_mode: str = "temperature"   # "temperature" (learnable T) or "sqrt_inv_d" (fixed 1/sqrt(d))
    sqrt_inv_d_dim: str = "intermediate"     # for sqrt_inv_d mode: "intermediate" (=intermediate_size) or "hidden" (=hidden_size)
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0.0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "BoltzRouter_TopK_Energy_MoE_MLP"
        assert self.n_experts >= 2
        assert 1 <= self.top_k <= self.n_experts
        assert self.boltzmann_temperature > 0
        assert self.energy_scale_mode in ("temperature", "sqrt_inv_d")
        assert self.sqrt_inv_d_dim in ("intermediate", "hidden")


class _LRDiagonalGaussBoltzmannMoEArgs(BaseArgs):
    mlp_type: str = "LRDiagonalGaussBoltzmannMoE"
    intermediate_size: int
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False
    num_experts: int = 32
    num_experts_per_tok: int = 2
    normalized_topk: bool = True
    use_det_normalization: bool = True       # include ½ log det(W^T W) in routing logits
    use_mixing_coefficients: bool = True     # learnable GMM mixing weights π_e
    diversity_lambda: float = 0.01           # cosine diversity penalty on W matrices
    entropy_bonus_gamma: float = 0.0         # per-token routing entropy bonus
    mixing_entropy_gamma: float = 0.0        # entropy bonus on π_e mixing coefficients
    centroid_repulsion_lambda: float = 0.0   # pairwise centroid repulsion penalty
    bias_based_balancing: bool = False        # DeepSeek-V3 aux-loss-free balancing via per-expert bias
    bias_update_alpha: float = 0.001          # step size for bias adjustment

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "LRDiagonalGaussBoltzmannMoE"
