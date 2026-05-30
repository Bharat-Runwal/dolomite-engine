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
    top_k: int | None = None    # None = soft (all experts active); int = sparse top-k Boltzmann routing
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0.0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "BoltzmannMoE_Energy_MLP"
        assert self.n_experts >= 2, "BoltzmannMoE requires at least 2 experts"
        assert self.intermediate_size % self.n_experts == 0, (
            f"intermediate_size ({self.intermediate_size}) must be divisible by "
            f"n_experts ({self.n_experts})"
        )
        assert self.temperature > 0, "temperature must be positive"


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
