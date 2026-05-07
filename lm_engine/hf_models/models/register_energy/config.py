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

    def __init__(self, n_registers: int = 128, register_generation_mode: str = "bypass", **kwargs):
        super().__init__(**kwargs)
        self.n_registers = n_registers
        self.register_generation_mode = register_generation_mode
