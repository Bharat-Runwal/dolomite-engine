# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...config import CommonConfig
from .mlp import (
    MLP,
    Energy_MLP,
    Hopfield_Energy_MLP,
    BoltzmannMoE_Hopfield_Energy_MLP,
    Compositional_Energy_MLP,
    Mixed_Energy_MLP,
    BoltzmannMoE_Energy_MLP,
    TopK_Energy_MoE_MLP,
    SurrogateBoltzmannMoE_Energy_MLP,
    interleave_up_gate_tensor_for_mlp,
    split_up_gate_tensor_for_mlp,
)
from .energy_ff import (
    FFEnergyBase,
    W1W2FFEnergy,
    HopfieldFFEnergy,
    BoltzmannMoEFFEnergy,
    FusedMoEContainer,
    build_boltzmann_moe,
)
from .moe import MoE, ParameterizedExperts



def get_mlp_block(config: CommonConfig, use_padding_free_transformer: bool, layer_idx: int) -> MLP | MoE | Energy_MLP:
    block = config.mlp_blocks[layer_idx]
    mlp_type = block.mlp_type

    kwargs = dict(
        hidden_size=config.hidden_size,
        intermediate_size=block.intermediate_size,
        activation_function=block.activation_function,
        add_bias=block.add_bias,
        dropout=block.dropout,
        init_method=config.init_method,
        initializer_range=config.initializer_range,
        m_width=config.m_width,
        num_layers=config.num_layers,
    )

    if mlp_type == "MLP":
        mlp = MLP(**kwargs)
    
    elif mlp_type == "Energy_MLP":
        mlp = Energy_MLP(**kwargs)

    elif mlp_type == "Hopfield_Energy_MLP":
        # Bounded-below FFN energy: E_FF = ||gelu(Wh)||^2. Single shared weight W;
        # half the per-block FFN params of Energy_MLP, so double intermediate_size
        # at config time to keep param count matched.
        mlp = Hopfield_Energy_MLP(**kwargs)

    elif mlp_type == "BoltzmannMoE_Hopfield_Energy_MLP":
        # Boltzmann-MoE over Hopfield single-W experts. K experts, each with
        # expert_I = intermediate_size // n_experts neurons. E_k = (1/expert_I)
        # ||gelu(W_k h)||^2 ≥ 0; routing softmax(-E_k); E_total = -LSE_k(-E_k).
        # Iso-param with Hopfield_Energy_MLP at the same intermediate_size.
        mlp = BoltzmannMoE_Hopfield_Energy_MLP(
            hidden_size=config.hidden_size,
            intermediate_size=block.intermediate_size,
            n_experts=block.n_experts,
            activation_function=block.activation_function,
            add_bias=block.add_bias,
            dropout=block.dropout,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
        )

    elif mlp_type == "Mixed_Energy_MLP":
        mlp = Mixed_Energy_MLP(
            hidden_size=config.hidden_size,
            energy_intermediate_size=block.energy_intermediate_size,
            standard_intermediate_size=block.standard_intermediate_size,
            activation_function=block.activation_function,
            add_bias=block.add_bias,
            dropout=block.dropout,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
        )

    elif mlp_type == "Compositional_Energy_MLP":
        mlp = Compositional_Energy_MLP(
            **kwargs,
            num_paths=block.num_paths,
            path_activations=block.path_activations,
            layer_idx=layer_idx,
        )

    elif mlp_type == "MoE":
        mlp = MoE(
            **kwargs,
            shared_intermediate_size=block.shared_intermediate_size,
            use_interleaved_weights=block.use_interleaved_weights,
            shared_expert_gating=block.shared_expert_gating,
            normalized_topk=block.normalized_topk,
            num_experts=block.num_experts,
            num_experts_per_tok=block.num_experts_per_tok,
            use_padding_free_transformer=use_padding_free_transformer,
        )

    elif mlp_type == "BoltzmannMoE_Energy_MLP":
        mlp = BoltzmannMoE_Energy_MLP(
            hidden_size=config.hidden_size,
            intermediate_size=block.intermediate_size,
            n_experts=block.n_experts,
            temperature=block.temperature,
            repulsion_coef=block.repulsion_coef,
            n_repulsion_pairs=block.n_repulsion_pairs,
            top_k=block.top_k,
            gelu_grad_method=getattr(block, "gelu_grad_method", "sigmoid"),
            activation_function=block.activation_function,
            add_bias=block.add_bias,
            dropout=block.dropout,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
        )

    elif mlp_type == "TopK_Energy_MoE_MLP":
        mlp = TopK_Energy_MoE_MLP(
            hidden_size=config.hidden_size,
            intermediate_size=block.intermediate_size,
            n_experts=block.n_experts,
            top_k=block.top_k,
            load_balance_coef=block.load_balance_coef,
            activation_function=block.activation_function,
            add_bias=block.add_bias,
            dropout=block.dropout,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
        )

    elif mlp_type == "EnergyFF_W1W2":
        # 2026-06-28 refactor: composable W1W2 energy FF. Drop-in for Energy_MLP.
        mlp = W1W2FFEnergy(
            hidden_size=config.hidden_size,
            intermediate_size=block.intermediate_size,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            add_bias=block.add_bias,
            gelu_grad_method=getattr(block, "gelu_grad_method", "sigmoid"),
            layer_idx=layer_idx,
        )

    elif mlp_type == "EnergyFF_Hopfield":
        # 2026-06-28 refactor: composable Hopfield energy FF. Drop-in for
        # Hopfield_Energy_MLP. Single shared W; E = (1/d_int)||gelu(Wh)||² ≥ 0.
        mlp = HopfieldFFEnergy(
            hidden_size=config.hidden_size,
            intermediate_size=block.intermediate_size,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            add_bias=block.add_bias,
            gelu_grad_method=getattr(block, "gelu_grad_method", "sigmoid"),
            layer_idx=layer_idx,
        )

    elif mlp_type == "EnergyFF_BoltzmannMoE":
        # 2026-06-28 refactor: composable Boltzmann-MoE over W1W2 OR Hopfield
        # experts. Restores repulsion + τ + n_repulsion_pairs that the
        # standalone BoltzmannMoE_Hopfield_Energy_MLP was missing.
        mlp = build_boltzmann_moe(
            expert_kind=block.expert_kind,
            hidden_size=config.hidden_size,
            intermediate_size=block.intermediate_size,
            n_experts=block.n_experts,
            temperature=block.temperature,
            repulsion_coef=block.repulsion_coef,
            n_repulsion_pairs=block.n_repulsion_pairs,
            top_k=block.top_k,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            add_bias=block.add_bias,
            gelu_grad_method=getattr(block, "gelu_grad_method", "sigmoid"),
            layer_idx=layer_idx,
        )

    elif mlp_type == "SurrogateBoltzmannMoE_Energy_MLP":
        mlp = SurrogateBoltzmannMoE_Energy_MLP(
            hidden_size=config.hidden_size,
            intermediate_size=block.intermediate_size,
            n_experts=block.n_experts,
            temperature=block.temperature,
            repulsion_coef=block.repulsion_coef,
            n_repulsion_pairs=block.n_repulsion_pairs,
            surrogate_coef=block.surrogate_coef,
            use_surrogate=block.use_surrogate,
            activation_function=block.activation_function,
            add_bias=block.add_bias,
            dropout=block.dropout,
            init_method=config.init_method,
            initializer_range=config.initializer_range,
            m_width=config.m_width,
            num_layers=config.num_layers,
            layer_idx=layer_idx,
        )

    else:
        raise ValueError(f"invalid mlp_type ({mlp_type}) for layer ({layer_idx})")

    return mlp
