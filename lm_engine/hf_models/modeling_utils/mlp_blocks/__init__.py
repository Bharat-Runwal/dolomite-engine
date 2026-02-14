# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...config import CommonConfig
from .mlp import MLP, interleave_up_gate_tensor_for_mlp, split_up_gate_tensor_for_mlp, Energy_MLP
from .moe import MoE, ParameterizedExperts
from .moe_energy import MoE_Energy, MoE_Energy_Module, MoE_Energy_F5, GaussianBoltzmannMoE



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
    elif mlp_type == "MoE_Energy":
        mlp = MoE_Energy(
            **kwargs,
            shared_intermediate_size=block.shared_intermediate_size,
            shared_expert_gating=block.shared_expert_gating,
            normalized_topk=block.normalized_topk,
            num_experts=block.num_experts,
            num_experts_per_tok=block.num_experts_per_tok,
            use_padding_free_transformer=use_padding_free_transformer,
            stop_gradient_routing=block.stop_gradient_routing,
        )
    elif mlp_type == "MoE_Energy_Module":
        mlp = MoE_Energy_Module(
            **kwargs,
            shared_intermediate_size=block.shared_intermediate_size,
            shared_expert_gating=block.shared_expert_gating,
            normalized_topk=block.normalized_topk,
            num_experts=block.num_experts,
            num_experts_per_tok=block.num_experts_per_tok,
            use_padding_free_transformer=use_padding_free_transformer,
            energy_routing=block.energy_routing,
            energy_routing_tau=block.energy_routing_tau,
        )
    elif mlp_type == "MoE_Energy_F5":
        mlp = MoE_Energy_F5(
            **kwargs,
            shared_intermediate_size=block.shared_intermediate_size,
            shared_expert_gating=block.shared_expert_gating,
            normalized_topk=block.normalized_topk,
            num_experts=block.num_experts,
            num_experts_per_tok=block.num_experts_per_tok,
            use_padding_free_transformer=use_padding_free_transformer,
            boltzmann_temperature=block.boltzmann_temperature,
            learnable_temperature=block.learnable_temperature,
            distillation_weight=block.distillation_weight,
            use_boltzmann_at_inference=block.use_boltzmann_at_inference,
        )
    elif mlp_type == "GaussianBoltzmannMoE":
        mlp = GaussianBoltzmannMoE(
            **kwargs,
            num_experts=block.num_experts,
            num_experts_per_tok=block.num_experts_per_tok,
            normalized_topk=block.normalized_topk,
            use_padding_free_transformer=use_padding_free_transformer,
            use_det_normalization=block.use_det_normalization,
            use_mixing_coefficients=block.use_mixing_coefficients,
            diversity_lambda=block.diversity_lambda,
            entropy_bonus_gamma=block.entropy_bonus_gamma,
            mixing_entropy_gamma=block.mixing_entropy_gamma,
            centroid_repulsion_lambda=block.centroid_repulsion_lambda,
            bias_based_balancing=block.bias_based_balancing,
            bias_update_alpha=block.bias_update_alpha,
            kl_distillation=block.kl_distillation,
            distillation_weight=block.distillation_weight,
            use_gmm_at_inference=block.use_gmm_at_inference,
        )
    else:
        raise ValueError(f"invalid mlp_type ({mlp_type}) for layer ({layer_idx})")

    return mlp
