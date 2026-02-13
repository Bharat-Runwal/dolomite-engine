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

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE_Energy_F5"


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

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "GaussianBoltzmannMoE"
