# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ..energy.config import EnergyConfig


class RegisterEnergyConfig(EnergyConfig):
    """EnergyConfig extended with learnable register tokens.

    Register tokens are prepended to the hidden state sequence before the
    recurrent energy blocks and stripped off afterwards. They attend to all
    tokens via full attention (not masked), enabling global information routing
    without polluting the LM loss.

    New fields:
        n_registers: int  — number of learnable register tokens (default 128).
                            Set to 0 to disable (equivalent to plain EnergyModel).
    """

    model_type = "register_energy"

    def __init__(self, n_registers: int = 128, register_generation_mode: str = "bypass",
                 register_start_layer: int = 0,
                 register_attn_balance_coef: float = 0.0,
                 register_attn_balance_threshold: float = 0.5,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_registers = n_registers
        self.register_generation_mode = register_generation_mode
        # register_start_layer: first layer index at which registers are injected.
        # 0 = all layers (default, original behaviour).
        # >0 = registers only active from that layer onward — e.g. set to 6 for a
        # 6-GPT+1-EGPT×6 hybrid to add registers only to the energy block.
        self.register_start_layer = register_start_layer

        # Register attention-balance auxiliary loss: penalises the attention mass that
        # content queries put on register keys (the failure mode where registers
        # "hog" attention, observed at 94-96% on hop_r256). For each layer where
        # registers are active, mass_l = mean over (B, heads, content_query) of
        # sum_{r=0..R-1} attn_probs[..., :R]. Per-layer hinge max(0, mass_l - thr)
        # is summed across active layers and added to the total loss as
        # coef * sum_l max(0, mass_l - thr). coef=0 ⇒ totally inert.
        # Note: enabling this doubles attention compute for register-active layers
        # in training because we take the manual matmul path to read attn_probs.
        self.register_attn_balance_coef = register_attn_balance_coef
        self.register_attn_balance_threshold = register_attn_balance_threshold
