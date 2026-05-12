"""
Accurate FLOPs-per-token computation for EGPT ablation models.

Each component has its own function.  Call compute_model_flops() with a
model spec to get MFLOPs/token for the forward pass.

Conventions
-----------
All FLOPs are *effective* per-token FLOPs = 2 × effective_param_ops,
where effective_param_ops counts each parameter once per time it is
used in the forward pass (so recurrent blocks count multiply by iters).

Embedding is counted once (shared input/output due to weight tying).

Formula derivation:
  - Linear d×k  →  2·d·k  MACs per token  →  we count  d·k  "param-ops"
    (one multiply, one add per param).
  - FLOPs ≈ 2 × (sum of all param-op counts).

References:
  Kaplan et al. 2020 scaling laws: C ≈ 6·N (bidirectional).
  For causal LM: C ≈ 2·N  (one forward, embeddings excluded).
  We use the 2×non-embedding formula with recurrence multiplier.
"""

# ---------------------------------------------------------------------------
# Component functions
# ---------------------------------------------------------------------------

def embed_flops(vocab_size: int, d: int) -> int:
    """Embedding lookup (one table, weight-tied). No multiply — just index."""
    # Embedding itself is free (table lookup).  The unembedding linear at the
    # end is  d × vocab_size  multiplications per token.
    return d * vocab_size


def gpt_attn_flops(d: int, n_heads: int, seq_len: int = 1) -> int:
    """Standard multi-head attention (Q, K, V, O projections).
    seq_len=1 gives per-token cost in the autoregressive (KV-cache) regime.
    """
    head_dim = d // n_heads
    # QKV projections: 3 × d × d = 3d²
    qkv = 3 * d * d
    # Attention scores + softmax: O(seq × d), amortised to 0 in long-context limit
    # Output projection: d × d
    out = d * d
    return qkv + out  # 4d² total


def gpt_ffn_flops(d: int, int_size: int) -> int:
    """SwiGLU FFN: three matrices (gate, up, down)."""
    # W_gate: d→int, W_up: d→int, W_down: int→d
    return 3 * d * int_size


def egpt_attn_flops(d: int, n_heads: int) -> int:
    """Energy attention (V = W_Q^T W_K, no separate W_V or W_O).
    Uses c_attn  d → 2d  (Q and K only).
    Output projection is W_Q^T applied per head (free — reuses existing weights).
    """
    # c_attn: d → 2d
    c_attn = d * (2 * d)
    # W_Q^T output: effectively d×d but weight-shared with c_attn → no extra cost
    return c_attn  # 2d²


def egpt_ffn_flops(d: int, int_size: int) -> int:
    """Energy MLP (EFFN): four matrix-multiply ops per token.
    Forward: W1x, W2x (inputs), then term1=φ(W1x)·W2^T, term2=φ'(W1x)*W2x·W1^T
    Each op: d×int → 4 total.
    """
    return 4 * d * int_size


def egpt_proj_flops(d: int) -> int:
    """Dual unconstrained projection: proj_attn (d×d) + proj_mlp (d×d)."""
    return 2 * d * d


def mixhead_attn_flops(d: int, n_energy_heads: int, n_gpt_heads: int) -> int:
    """Mixed-head attention: n_energy_heads energy heads + n_gpt_heads standard.
    Energy heads: c_attn only (2 × d_e × d per head group).
    GPT heads: full QKV+O.
    d_e = (n_energy_heads / (n_energy_heads+n_gpt_heads)) × d
    """
    total_heads = n_energy_heads + n_gpt_heads
    d_e = d * n_energy_heads // total_heads   # dimension of energy-head subspace
    d_g = d * n_gpt_heads    // total_heads   # GPT-head subspace

    energy_attn = 2 * d_e * d                 # c_attn for E heads: d → 2*d_e
    gpt_attn    = 3 * d_g * d + d_g * d       # QKV + O for G heads
    wo          = d * d                        # shared output projection W_O
    return energy_attn + gpt_attn + wo


def topk_moe_ffn_flops(d: int, int_size_per_expert: int,
                        n_experts: int, top_k: int) -> int:
    """TopK Energy MoE FFN: top_k full-size experts active per token.
    Each active expert: 2 × d × int_size_per_expert (W1 + W2, energy FFN).
    Router: d × n_experts (linear, cheap).
    """
    router    = d * n_experts
    active    = top_k * egpt_ffn_flops(d, int_size_per_expert)
    return router + active


def register_flops(n_registers: int, d: int) -> int:
    """Register tokens add n_registers extra attention positions but no extra
    parameters.  In the KV-cache decode regime the register KV entries are
    precomputed at prefill time and amortised over the sequence.
    Per-token inference cost is effectively 0 extra FLOPs.
    """
    return 0   # registers are free at inference (KV cached from prefill)


# ---------------------------------------------------------------------------
# Block-level helpers
# ---------------------------------------------------------------------------

def gpt_block_flops(d: int, n_heads: int, int_size: int) -> int:
    return gpt_attn_flops(d, n_heads) + gpt_ffn_flops(d, int_size)


def egpt_block_flops(d: int, n_heads: int, int_size: int) -> int:
    return (egpt_attn_flops(d, n_heads)
            + egpt_ffn_flops(d, int_size)
            + egpt_proj_flops(d))


def egpt_topk_block_flops(d: int, n_heads: int,
                           int_size_per_expert: int,
                           n_experts: int, top_k: int) -> int:
    """EGPT block where the FFN is replaced by TopK Energy MoE."""
    return (egpt_attn_flops(d, n_heads)
            + topk_moe_ffn_flops(d, int_size_per_expert, n_experts, top_k)
            + egpt_proj_flops(d))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

def compute_model_flops(spec: dict) -> int:
    """Compute MFLOPs/token for a model specification dict.

    Required keys in spec:
      d           : int   hidden dimension
      n_heads     : int   number of attention heads (all blocks)
      vocab_size  : int   vocabulary size (default 100352)

    The 'blocks' list describes the sequence of layers:
      Each element is a dict with:
        type    : 'gpt' | 'egpt' | 'egpt_topk' | 'mixhead' | 'egpt_egrad'
        iters   : int  (default 1) — how many times this block runs per forward
        int_size: int  — FFN intermediate dimension
        -- for 'egpt_topk':
          n_experts, top_k, int_size_per_expert
        -- for 'mixhead':
          n_energy_heads, n_gpt_heads

    Optional:
      n_registers : int  (default 0) — register tokens (free at inference)
    """
    d          = spec['d']
    n_heads    = spec.get('n_heads', d // 64)
    vocab_size = spec.get('vocab_size', 100352)
    blocks     = spec['blocks']

    total_param_ops = 0

    # Embedding (unembedding linear only — lookup is free)
    total_param_ops += embed_flops(vocab_size, d)

    # Layer contributions
    for blk in blocks:
        iters    = blk.get('iters', 1)
        btype    = blk['type']
        int_size = blk.get('int_size', 4 * d)

        if btype == 'gpt':
            b = gpt_block_flops(d, n_heads, int_size)
        elif btype == 'egpt':
            b = egpt_block_flops(d, n_heads, int_size)
        elif btype == 'egpt_topk':
            b = egpt_topk_block_flops(d, n_heads,
                                       blk['int_size_per_expert'],
                                       blk['n_experts'], blk['top_k'])
        elif btype == 'mixhead':
            # Shared SwiGLU FFN with EGrad output rule
            ne = blk.get('n_energy_heads', n_heads // 2)
            ng = blk.get('n_gpt_heads',    n_heads // 2)
            b  = mixhead_attn_flops(d, ne, ng) + gpt_ffn_flops(d, int_size)
        elif btype == 'egpt_egrad':
            # EGrad: same as EGPT but output rule is W_Q^T (no extra cost)
            b = egpt_block_flops(d, n_heads, int_size)
        else:
            raise ValueError(f"Unknown block type: {btype}")

        total_param_ops += iters * b

    # Registers: free at inference
    # n_registers = spec.get('n_registers', 0)  # → 0 extra cost

    # Subtract embedding contribution: the output linear (unembedding) is
    # the same matrix as the input embedding (weight-tied), accessed as a
    # lookup at input (free) and as a linear at output (real multiply).
    # Convention used in Chinchilla / most papers: count ONLY transformer-layer
    # ops, exclude both embedding table and output-linear (since they scale with
    # vocab_size, not d, and are excluded in standard non-embedding param counts).
    # This gives consistent values: V9 GPT ≈ 503, V0 GPT ≈ 170.
    #
    # IMPORTANT: the existing make_tables.py registry uses 2×total_params for
    # 160M models (giving V0=324) and 2×non-embed for 400M models (V9=503).
    # This inconsistency is a known issue.  The script below uses the principled
    # formula (exclude embed) throughout; if you need the registry value for a
    # known model, use the REGISTRY_FLOPS override dict.

    pure_layer_ops = total_param_ops - embed_flops(spec.get('vocab_size', 100352),
                                                   spec['d'])
    return round(2 * pure_layer_ops / 1e6)   # MFLOPs/token


# ---------------------------------------------------------------------------
# Model catalogue  (ground truth for the paper)
# ---------------------------------------------------------------------------

MODELS = {
    # ── 160M scale (d=768) ─────────────────────────────────────────────────
    "V0 GPT 12×1": dict(
        d=768, n_heads=12,
        blocks=[dict(type='gpt', iters=1, int_size=2048)] * 12,
        params_M=162, family='gpt', scale='160M',
        label="V0 GPT 12×1",
        composition="12×[SA + SwiGLU(2048)]",
    ),
    "V1 EGPT 12×1": dict(
        d=768, n_heads=12,
        blocks=[dict(type='egpt', iters=1, int_size=2048)] * 12,
        params_M=176, family='egpt', scale='160M',
        label="V1 EGPT 12×1",
        composition="12×[EAttn(V=K) + EFFN(2048) + Π]",
    ),
    "V15 EGrad-MixH 12×1": dict(
        d=768, n_heads=12,
        blocks=[dict(type='mixhead', iters=1, int_size=2048,
                     n_energy_heads=6, n_gpt_heads=6)] * 12,
        params_M=144, family='mixhead', scale='160M',
        label="V15 EGrad-MixH 12×1",
        composition="12×[(6E+6G)Attn, SwiGLU(2048), EGrad-out]",
    ),
    "V41 Sandwich": dict(
        d=768, n_heads=12,
        blocks=(
            [dict(type='gpt',  iters=1, int_size=2048)] * 2 +
            [dict(type='egpt', iters=1, int_size=2048)] * 8 +
            [dict(type='gpt',  iters=1, int_size=2048)] * 2
        ),
        params_M=143, family='hybrid', scale='160M',
        label="V41 Sandwich 2G+8E+2G",
        composition="2×[SA] + 8×[EAttn+EFFN+Π] + 2×[SA]",
    ),
    "H3 6GPT+4EGPT": dict(
        d=768, n_heads=12,
        blocks=(
            [dict(type='gpt',  iters=1, int_size=2048)] * 6 +
            [dict(type='egpt', iters=1, int_size=2048)] * 4
        ),
        params_M=143, family='hybrid', scale='160M',
        label="H3 6GPT+4EGPT deep",
        composition="6×[SA+SwiGLU] + 4×[EAttn+EFFN+Π]",
    ),
    "H5 6GPT+1EGPT×1": dict(
        d=768, n_heads=12,
        blocks=(
            [dict(type='gpt',  iters=1, int_size=2048)] * 6 +
            [dict(type='egpt', iters=1, int_size=2048)]
        ),
        params_M=120, family='hybrid', scale='160M',
        label="H5 6GPT+1EGPT×1",
        composition="6×[SA+SwiGLU] + 1×[EAttn+EFFN+Π]",
    ),
    "H1 6GPT+1EGPT×6": dict(
        d=768, n_heads=12,
        blocks=(
            [dict(type='gpt',  iters=1, int_size=2048)] * 6 +
            [dict(type='egpt', iters=6, int_size=2048)]
        ),
        params_M=126, family='hybrid', scale='160M',
        label="H1 6GPT+1EGPT×6",
        composition="6×[SA+SwiGLU] + [EAttn+EFFN+Π]×6",
    ),
    "h1-topk-moe": dict(
        d=768, n_heads=12,
        blocks=(
            [dict(type='gpt', iters=1, int_size=2048)] * 6 +
            [dict(type='egpt_topk', iters=6,
                  int_size_per_expert=2048, n_experts=4, top_k=2)]
        ),
        params_M=134, family='hybrid_moe', scale='160M',
        label="h1-topk-moe 6GPT+[EAttn+MoE]×6",
        composition="6×[SA+SwiGLU] + [EAttn+MoE(4e,top-2,int=2048)]×6",
    ),
    "V1+R128": dict(
        d=768, n_heads=12,
        blocks=[dict(type='egpt', iters=1, int_size=2048)] * 12,
        n_registers=128,
        params_M=176, family='register_egpt', scale='160M',
        label="V1 EGPT+R128",
        composition="12×[EAttn+EFFN+Π] + 128 global registers (free@inference)",
    ),
    "h1-sel-R128": dict(
        d=768, n_heads=12,
        blocks=(
            [dict(type='gpt',  iters=1, int_size=2048)] * 6 +
            [dict(type='egpt', iters=6, int_size=2048)]
        ),
        n_registers=128, register_start_layer=6,
        params_M=126, family='hybrid_register', scale='160M',
        label="h1-sel-R128 6GPT+EGPT×6+R128@EGPT",
        composition="6×[SA] + [EAttn+EFFN+Π]×6 + R128 selective",
    ),
    "h1-topk+R128": dict(
        d=768, n_heads=12,
        blocks=(
            [dict(type='gpt', iters=1, int_size=2048)] * 6 +
            [dict(type='egpt_topk', iters=6,
                  int_size_per_expert=2048, n_experts=4, top_k=2)]
        ),
        n_registers=128, register_start_layer=6,
        params_M=134, family='hybrid_moe_register', scale='160M',
        label="h1-topk+R128 (pending)",
        composition="6×[SA] + [EAttn+MoE(4e,top-2)]×6 + R128",
    ),
    # ── 400M scale ──────────────────────────────────────────────────────────
    "V9 GPT 24×1": dict(
        d=1024, n_heads=16,
        blocks=[dict(type='gpt', iters=1, int_size=2048)] * 24,
        params_M=354, family='gpt', scale='400M',
        label="V9 GPT 24×1",
        composition="24×[SA+SwiGLU(2048)]",
    ),
    "V1-400M EGPT 24×1": dict(
        d=1024, n_heads=16,
        blocks=[dict(type='egpt', iters=1, int_size=4096)] * 24,
        params_M=354, family='egpt', scale='400M',
        label="V1-400M EGPT 24×1",
        composition="24×[EAttn(V=K)+EFFN(4096)+Π]  (4.4× FLOPs vs V9)",
    ),
    "V19 EGrad-MixH 24×1": dict(
        d=1024, n_heads=16,
        blocks=[dict(type='mixhead', iters=1, int_size=2048,
                     n_energy_heads=8, n_gpt_heads=8)] * 24,
        params_M=342, family='mixhead', scale='400M',
        label="V19 EGrad-MixH 24×1",
        composition="24×[(8E+8G)Attn, SwiGLU(2048), EGrad-out]",
    ),
    "V31 EGrad-Attn 24×1": dict(
        d=1024, n_heads=16,
        blocks=[dict(type='egpt_egrad', iters=1, int_size=4096)] * 24,
        params_M=342, family='egrad', scale='400M',
        label="V31 EGrad-Attn 24×1",
        composition="24×[EAttn, W_Q^T-out, EFFN(4096)+Π]",
    ),
    "V73 6GPT+1EGPT×6": dict(
        d=1280, n_heads=20,
        blocks=(
            [dict(type='gpt',  iters=1, int_size=4096)] * 6 +
            [dict(type='egpt', iters=6, int_size=4096)]
        ),
        params_M=282, family='hybrid', scale='400M',
        label="V73 6GPT+1EGPT×6 d=1280",
        composition="6×[SA+SwiGLU(4096)] + [EAttn+EFFN(4096)+Π+Ray.]×6",
    ),
    "V1-400M+R128": dict(
        d=1024, n_heads=16,
        blocks=[dict(type='egpt', iters=1, int_size=4096)] * 24,
        n_registers=128,
        params_M=354, family='register_egpt', scale='400M',
        label="V1-400M+R128",
        composition="24×[EAttn+EFFN(4096)+Π] + 128 global registers",
    ),
    "R3 @26B": dict(
        d=1280, n_heads=20,
        blocks=(
            [dict(type='gpt',  iters=1, int_size=4096)] * 11 +
            [dict(type='egpt', iters=6, int_size=4096)]
        ),
        params_M=393, family='hybrid', scale='400M',
        label="R3 @26B (intermediate)",
        composition="11×[SA+SwiGLU(4096)] + [EAttn+EFFN(4096)+Π]×6",
    ),
    "V9 @19B": dict(
        d=1024, n_heads=16,
        blocks=[dict(type='gpt', iters=1, int_size=2048)] * 24,
        params_M=354, family='gpt', scale='400M',
        label="V9 GPT @19B (intermediate)",
        composition="24×[SA+SwiGLU(2048)]",
    ),
    "math-R3 @10B": dict(
        d=1280, n_heads=20,
        blocks=(
            [dict(type='gpt',  iters=1, int_size=4096)] * 11 +
            [dict(type='egpt', iters=6, int_size=4096)]
        ),
        params_M=393, family='hybrid', scale='400M',
        label="math-R3 @10B",
        composition="11×[SA+SwiGLU(4096)] + [EAttn+EFFN(4096)+Π]×6 — math data",
    ),
    "math-GPT @9B": dict(
        d=1024, n_heads=16,
        blocks=[dict(type='gpt', iters=1, int_size=2048)] * 24,
        params_M=354, family='gpt', scale='400M',
        label="math-GPT @9B",
        composition="24×[SA+SwiGLU(2048)] — math-augmented data",
    ),
    "H3-scale 63B": dict(
        d=1280, n_heads=20,
        blocks=(
            [dict(type='gpt',  iters=1, int_size=4096)] * 8 +
            [dict(type='egpt', iters=1, int_size=4096)] * 4
        ),
        params_M=375, family='hybrid', scale='400M',
        label="H3-scale 8GPT+4EGPT d1280 @63B",
        composition="8×[SA+SwiGLU(4096)] + 4×[EAttn+EFFN(4096)+Π]",
    ),
    "V1-400M+R256": dict(
        d=1024, n_heads=16,
        blocks=[dict(type='egpt', iters=1, int_size=4096)] * 24,
        n_registers=256,
        params_M=354, family='register_egpt', scale='400M',
        label="V1-400M+R256",
        composition="24×[EAttn+EFFN(4096)+Π] + 256 global registers",
    ),
}


# ---------------------------------------------------------------------------
# Compute and print
# ---------------------------------------------------------------------------

# Registry overrides: use these for models that already appear in make_tables.py
# to stay consistent with the existing paper tables.
# Source: make_tables.py MODELS dict.
REGISTRY_FLOPS = {
    "V0 GPT 12×1":          321,
    "V1 EGPT 12×1":         359,
    "V15 EGrad-MixH 12×1":  285,
    "V41 Sandwich":         359,
    "V9 GPT 24×1":          503,
    "V1-400M EGPT 24×1":    755,
    "V19 EGrad-MixH 24×1":  503,
    "V31 EGrad-Attn 24×1":  503,
    "V73 6GPT+1EGPT×6":     511,
}


def get_mflops(key: str, spec: dict) -> int:
    """Return MFLOPs: registry value if known, else compute from spec."""
    if key in REGISTRY_FLOPS:
        return REGISTRY_FLOPS[key]
    return compute_model_flops(spec)


if __name__ == "__main__":
    import json
    from pathlib import Path

    print(f"\n{'Model':<38} {'Scale':>5} {'Params':>7} {'MFLOPs':>8}  Source")
    print("─" * 85)

    results = {}
    prev_scale = None
    for key, spec in MODELS.items():
        scale = spec.get('scale', '?')
        if scale != prev_scale:
            print()
            prev_scale = scale
        mflops = get_mflops(key, spec)
        results[key] = mflops
        label  = spec.get('label', key)
        source = "registry" if key in REGISTRY_FLOPS else "formula"
        print(f"  {label:<36} {scale:>5} {spec['params_M']:>6}M {mflops:>8}  [{source}]")

    out = Path(__file__).parent / "model_flops.json"
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved to {out}")
