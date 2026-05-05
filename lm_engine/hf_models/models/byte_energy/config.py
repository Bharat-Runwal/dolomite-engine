from ..energy.config import EnergyConfig


class ByteEnergyConfig(EnergyConfig):
    """Energy model with w4s2 byte-level tokenizer instead of BPE embedding.

    Replaces the large BPE embedding (vocab_size × D params) with:
    - ByteLinearPool: Embedding(256, d_local) + Linear(W×d_local, D)  [tiny params]
    - ByteDecoder:    Linear(D, S×D) → Linear(D, 256)                 [tiny params]

    This frees ~77–160M params (previously in BPE embedding at 100k vocab) to go
    entirely into transformer computation.

    Key additions vs EnergyConfig:
      d_local     (int):  Byte embedding dim before linear pool (default 64)
      window_size (int):  Bytes per compressed token, w in wWsS (default 4)
      stride      (int):  Stride of linear pool, s in wWsS (default 2)

    Compression: input_seq_len → input_seq_len // stride compressed tokens.
    vocab_size should be set to 256 (byte alphabet).
    """
    model_type = "byte_energy"

    def __init__(self, d_local: int = 64, window_size: int = 4, stride: int = 2, **kwargs):
        # Force byte vocabulary
        kwargs.setdefault("vocab_size", 256)
        super().__init__(**kwargs)
        self.d_local = d_local
        self.window_size = window_size
        self.stride = stride
