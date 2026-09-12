"""Design + cost single-block recurrent all-Boltzmann-MoE energy models.

Motivation
----------
The 8-GPT + 1-EGPT hybrid is the wrong denominator for a FLOPs claim: the GPT
prefix (37.5%) and the lm_head (25.5%) swamp whatever the MoE saves, capping the
end-to-end win at ~22%. The architecture that is faithful to the energy picture
puts **all** parameters in ONE energy block and iterates it T times. Then the MoE
*is* the model, and every FLOP saved in routing is a FLOP saved end-to-end (modulo
the lm_head, which is fixed by vocab).

This script costs that family analytically and emits training configs.

Key cost facts (derived, both expert kinds)
-------------------------------------------
Per iteration, per token, MACs, with I = total expert width, K experts, top-k:

  hopfield  E_k = (1/I_e)||gelu(W_k h)||^2      W_k: [I_e, d]
      first  matmul  W h        : d*I     <- yields every E_k AND the pre-acts
      second matmul  gated @ W  : d*I     <- only needed for SELECTED experts
      dense soft                : 2*d*I

  w1w2      E_k = -(1/sqrt(I_e)) phi(W1_k h).(W2_k h)     W1_k,W2_k: [I_e, d]
      W1 h                      : d*I
      term1 = phi @ W2          : d*I     <- needed to form E_k, ALSO part of the output
      W2 h                      : d*I
      term2 = (phi' * W2h) @ W1 : d*I
      dense soft                : 4*d*I

In BOTH forms the router's work is fully reused by the experts it selects; the only
waste is the router work spent on the K-k *rejected* experts, which is exactly half
the total. Hence both forms have the SAME relative savings:

      exact energy router + top-k : 0.5 + 0.5*k/K   of dense
      free  proxy router + top-k :        k/K       of dense

w1w2 costs 2x hopfield at equal I, so at iso-parameters (I_w1w2 = I_hopfield/2)
they are also iso-FLOP. The cheap-proxy router is worth the same fraction in both.

Attention capacity in a single block
------------------------------------
With one shared block you cannot buy attention capacity by adding layers, so the
FFN:Attn imbalance that sank the B-series (21:1) reappears by construction. The
remedy is over-complete heads: let num_heads * head_dim = D_qk > d. That is a
free knob in the cost model, but NOT yet supported by the code --
``EnergyAttention_QK`` hard-couples ``head_dim = divide_if_divisible(hidden_size,
num_heads)`` (energy_attention.py:75). See --emit for which configs need it.

Usage
-----
  python experiments/boltzmann-moe/scripts/design_single_block_moe_20260912.py
  python .../design_single_block_moe_20260912.py --emit          # write YAML configs
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO / "configs/single_block_moe"

VOCAB = 100352
SEQ = 4096


@dataclass
class Arch:
    name: str
    d: int
    n_iter: int                  # T: how many times the single block is applied
    K: int                       # number of experts
    top_k: int
    expert_kind: str = "hopfield"      # "hopfield" | "w1w2"
    I_tot: int = 0                     # total expert width (0 => solve for target_params)
    num_heads: int = 16
    head_dim: int = 128
    proj_rank: int = 0                 # 0 => full (d); S is [d, proj_rank]
    antisym_rank: int = 32
    target_params: float | None = None  # if set and I_tot==0, solve I_tot for this
    note: str = ""

    # ---- derived ----------------------------------------------------------
    @property
    def D_qk(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def overcomplete(self) -> float:
        return self.D_qk / self.d

    @property
    def rank_s(self) -> int:
        return self.proj_rank if self.proj_rank > 0 else self.d

    def attn_params(self) -> int:
        # c_attn -> (Q,K) only; V=K and the output projection reuses the Q weights
        return 2 * self.D_qk * self.d

    def proj_params(self) -> int:
        return self.d * self.rank_s + 2 * self.d * self.antisym_rank

    def moe_params(self) -> int:
        mult = 1 if self.expert_kind == "hopfield" else 2
        return mult * self.I_tot * self.d

    def embed_params(self) -> int:
        return VOCAB * self.d          # tied

    def solve_I(self) -> None:
        if self.I_tot or self.target_params is None:
            return
        mult = 1 if self.expert_kind == "hopfield" else 2
        budget = self.target_params - self.embed_params() - self.attn_params() - self.proj_params()
        I = int(budget / (mult * self.d))
        # keep divisible by K and a multiple of 128 per expert for clean GEMMs
        per = max(128, (I // self.K) // 128 * 128)
        self.I_tot = per * self.K

    def total_params(self) -> int:
        return (self.embed_params() + self.attn_params()
                + self.proj_params() + self.moe_params())

    def active_params(self) -> int:
        """Params touched per token: everything except the K-k rejected experts."""
        mult = 1 if self.expert_kind == "hopfield" else 2
        active_moe = mult * (self.I_tot // self.K) * self.top_k * self.d
        return (self.embed_params() + self.attn_params()
                + self.proj_params() + active_moe)

    # ---- FLOPs (MACs per token; FLOPs = 2x) -------------------------------
    def macs(self, router: str, seq: int = SEQ) -> dict:
        """router in {"dense", "exact", "free"}. Causal attention => L/2 average."""
        mult = 1 if self.expert_kind == "hopfield" else 2
        dI = self.d * self.I_tot
        base = (2 if mult == 1 else 4) * dI                # dense soft routing
        if router == "dense":
            moe = base
        elif router == "exact":
            moe = int(base * (0.5 + 0.5 * self.top_k / self.K))
        elif router == "free":
            moe = int(base * self.top_k / self.K)
        else:
            raise ValueError(router)

        attn_lin = 3 * self.D_qk * self.d                 # c_attn (2x) + out-proj (1x)
        # scores Q@K^T and AV are each H*Dh*L MACs/token => 2*H*Dh*L, halved by
        # causal masking on average => H*Dh*L.
        attn_sdp = self.num_heads * self.head_dim * seq
        proj = 2 * self.d * self.rank_s + 4 * self.d * self.antisym_rank
        head = VOCAB * self.d

        per_iter = attn_lin + attn_sdp + proj + moe
        return {
            "moe": self.n_iter * moe,
            "attn_lin": self.n_iter * attn_lin,
            "attn_sdp": self.n_iter * attn_sdp,
            "proj": self.n_iter * proj,
            "lm_head": head,
            "total": self.n_iter * per_iter + head,
        }


def dense_twin(a: Arch) -> Arch:
    """Iso-ACTIVE-FLOPs dense (K=1) twin: the width a non-MoE EGPT would need to
    run at the same cost as ``a`` under a free router + top-k."""
    t = Arch(name=a.name + "__dense_twin", d=a.d, n_iter=a.n_iter, K=1, top_k=1,
             expert_kind=a.expert_kind,
             I_tot=max(1, a.I_tot * a.top_k // a.K),
             num_heads=a.num_heads, head_dim=a.head_dim,
             proj_rank=a.proj_rank, antisym_rank=a.antisym_rank,
             note="iso-active-FLOPs dense baseline")
    return t


# --------------------------------------------------------------------------- #
# the candidate family                                                        #
# --------------------------------------------------------------------------- #


def family() -> list[Arch]:
    A = []
    # Reference points from production MoEs (total/active ratio):
    #   Mixtral 8x7B 47B/13B = 3.6x | Qwen3-30B-A3B 30/3.3 = 9x
    #   DeepSeek-V3 671/37 = 18x    | OLMoE 7B/1B = 7x
    # We target 8-16x, i.e. K/k in that range.
    A += [
        Arch("s3b_K32k4_d2048_h64x128", d=2048, n_iter=8, K=32, top_k=4,
             num_heads=64, head_dim=128, target_params=3e9,
             note="3B total / ~8x sparsity, 4x over-complete heads"),
        Arch("s7b_K64k8_d2560_h64x128", d=2560, n_iter=8, K=64, top_k=8,
             num_heads=64, head_dim=128, target_params=7e9,
             note="7B total / 8x sparsity"),
        Arch("s7b_K128k8_d2560_h64x128", d=2560, n_iter=8, K=128, top_k=8,
             num_heads=64, head_dim=128, target_params=7e9,
             note="7B total / 16x sparsity (DeepSeek-like K)"),
        Arch("s12b_K128k8_d3072_h64x128", d=3072, n_iter=10, K=128, top_k=8,
             num_heads=64, head_dim=128, target_params=12e9,
             note="12B total / 16x sparsity"),
        Arch("s30b_K256k8_d3584_h96x128", d=3584, n_iter=12, K=256, top_k=8,
             num_heads=96, head_dim=128, target_params=30e9,
             note="30B total / 32x sparsity, where MoE is the default choice"),
        # control: same budget, heads NOT over-complete (D_qk == d)
        Arch("s7b_K64k8_d2560_h20x128_iso", d=2560, n_iter=8, K=64, top_k=8,
             num_heads=20, head_dim=128, target_params=7e9,
             note="CONTROL: D_qk == d, isolates the over-complete-heads effect"),
        # control: w1w2 experts at iso-params
        Arch("s7b_K64k8_d2560_w1w2", d=2560, n_iter=8, K=64, top_k=8,
             expert_kind="w1w2", num_heads=64, head_dim=128, target_params=7e9,
             note="CONTROL: w1w2 experts (2 matrices) at iso-params"),
    ]
    for a in A:
        a.solve_I()
    return A


def conventional(total_params: float, d: int, sparsity: float | None) -> dict:
    """A normal (non-recurrent) transformer at iso-TOTAL-params, for reference.

    Every parameter is touched at most once per token, so MACs ~= non-embedding
    params (x top-k/K for the MoE FFN part) + lm_head. This is the baseline the
    single-block recurrent design must eventually beat, and it exposes the
    recurrence tax: applying one shared block T times costs Tx the FLOPs of
    spreading the same parameters over T distinct layers.
    """
    nonemb = total_params - VOCAB * d
    if sparsity is None:                       # dense
        return {"macs": int(nonemb + VOCAB * d), "active": int(total_params)}
    # assume ~90% of non-embedding params are the sparse FFN (typical for MoE LLMs)
    ffn, rest = 0.90 * nonemb, 0.10 * nonemb
    return {"macs": int(ffn / sparsity + rest + VOCAB * d),
            "active": int(ffn / sparsity + rest + VOCAB * d)}


def fmt(n: float) -> str:
    for u, s in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= u:
            return f"{n/u:.2f}{s}"
    return f"{n:.0f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--emit", action="store_true", help="write YAML configs")
    args = ap.parse_args()

    A = family()

    print("=" * 118)
    print("SINGLE-BLOCK RECURRENT ALL-BOLTZMANN-MoE ENERGY MODELS  "
          f"(vocab={VOCAB}, seq={args.seq}, tied embeddings)")
    print("=" * 118)
    print(f"{'name':32s} {'d':>5s} {'T':>3s} {'K':>4s} {'k':>3s} {'I_tot':>7s} "
          f"{'Dqk/d':>6s} {'total':>8s} {'active':>8s} {'tot/act':>8s} {'expert%':>8s}")
    for a in A:
        tp, ap_ = a.total_params(), a.active_params()
        print(f"{a.name:32s} {a.d:5d} {a.n_iter:3d} {a.K:4d} {a.top_k:3d} {a.I_tot:7d} "
              f"{a.overcomplete:6.2f} {fmt(tp):>8s} {fmt(ap_):>8s} {tp/ap_:8.2f} "
              f"{100*a.moe_params()/tp:7.1f}%")

    print("\n" + "=" * 118)
    print("PER-TOKEN MACs  (prefill; FLOPs = 2x MACs).  'free' = cheap proxy router")
    print("=" * 118)
    print(f"{'name':32s} {'dense':>9s} {'exact-tk':>9s} {'free-tk':>9s} "
          f"{'MoE%dense':>10s} {'free/dense':>11s} {'GFLOP/tok':>10s}")
    for a in A:
        md = a.macs("dense", args.seq)
        me = a.macs("exact", args.seq)
        mf = a.macs("free", args.seq)
        print(f"{a.name:32s} {fmt(md['total']):>9s} {fmt(me['total']):>9s} "
              f"{fmt(mf['total']):>9s} {100*md['moe']/md['total']:9.1f}% "
              f"{mf['total']/md['total']:11.3f} {2*mf['total']/1e9:10.3f}")

    print("\n  breakdown of the 7B/16x point (dense routing):")
    a = [x for x in A if x.name == "s7b_K128k8_d2560_h64x128"][0]
    md = a.macs("dense", args.seq)
    for k, v in md.items():
        if k != "total":
            print(f"    {k:12s} {fmt(v):>9s}  {100*v/md['total']:5.1f}%")
    print(f"    {'TOTAL':12s} {fmt(md['total']):>9s}")

    print("\n" + "=" * 118)
    print("vs ISO-ACTIVE-FLOPs DENSE EGPT  (what a non-MoE energy model buys at the same cost)")
    print("=" * 118)
    print(f"{'name':32s} {'MoE total':>10s} {'dense total':>12s} {'capacity x':>11s} "
          f"{'MoE MACs':>9s} {'dense MACs':>11s}")
    for a in A:
        t = dense_twin(a)
        mf = a.macs("free", args.seq)
        mt = t.macs("dense", args.seq)
        print(f"{a.name:32s} {fmt(a.total_params()):>10s} {fmt(t.total_params()):>12s} "
              f"{a.total_params()/t.total_params():10.2f}x {fmt(mf['total']):>9s} "
              f"{fmt(mt['total']):>11s}")
    print("\n  Read: at equal inference FLOPs, the sparse single-block MoE carries")
    print("  'capacity x' more total parameters than the dense energy model it replaces.")

    print("\n" + "=" * 118)
    print("THE RECURRENCE TAX  (vs a CONVENTIONAL non-recurrent transformer at iso-total-params)")
    print("=" * 118)
    print(f"{'name':32s} {'total':>8s} {'ours free-tk':>13s} {'conv dense':>11s} "
          f"{'conv MoE':>10s} {'tax vs dense':>13s} {'tax vs MoE':>11s}")
    for a in A:
        mf = a.macs("free", args.seq)["total"]
        cd = conventional(a.total_params(), a.d, None)["macs"]
        cm = conventional(a.total_params(), a.d,
                          a.total_params() / a.active_params())["macs"]
        print(f"{a.name:32s} {fmt(a.total_params()):>8s} {fmt(mf):>13s} {fmt(cd):>11s} "
              f"{fmt(cm):>10s} {mf/cd:12.2f}x {mf/cm:10.2f}x")
    print("\n  A single shared block applied T times costs T x the FLOPs of spreading the")
    print("  same parameters over T distinct layers. Sparse routing is what pays for that:")
    print("  top-k/K cancels the T tax when k/K ~ 1/T. Read the last two columns as the")
    print("  compute premium the energy/recurrence story has to justify on quality.")

    print("\n" + "=" * 118)
    print("T SWEEP for s7b_K128k8_d2560_h64x128  (iso-total-params; I_tot re-solved per T)")
    print("=" * 118)
    print(f"{'T':>3s} {'I_tot':>8s} {'expert_I':>9s} {'active':>9s} {'tot/act':>8s} "
          f"{'free-tk MACs':>13s} {'tax vs conv dense':>18s}")
    base = [x for x in A if x.name == "s7b_K128k8_d2560_h64x128"][0]
    for T in (1, 2, 4, 6, 8, 12, 16):
        t = Arch(f"T{T}", d=base.d, n_iter=T, K=base.K, top_k=base.top_k,
                 num_heads=base.num_heads, head_dim=base.head_dim,
                 target_params=base.total_params())
        t.solve_I()
        mf = t.macs("free", args.seq)["total"]
        cd = conventional(t.total_params(), t.d, None)["macs"]
        print(f"{T:3d} {t.I_tot:8d} {t.I_tot//t.K:9d} {fmt(t.active_params()):>9s} "
              f"{t.total_params()/t.active_params():8.2f} {fmt(mf):>13s} {mf/cd:17.2f}x")

    need = [a.name for a in A if a.overcomplete != 1.0]
    print(f"\n  NOTE: {len(need)}/{len(A)} configs use over-complete heads (D_qk != d), which")
    print("  needs the EnergyAttention_QK head_dim decoupling (energy_attention.py:75).")

    if args.emit:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        for a in A:
            p = CONFIG_DIR / f"{a.name}.yml"
            p.write_text(emit_yaml(a))
            print(f"  wrote {p.relative_to(REPO)}")


def emit_yaml(a: Arch) -> str:
    mlp = (f"""      - mlp_type: EnergyFF_BoltzmannMoE
        expert_kind: {a.expert_kind}
        intermediate_size: {a.I_tot}
        n_experts: {a.K}
        top_k: {a.top_k}
        temperature: 1.0
        repulsion_coef: 0.01
        n_repulsion_pairs: 4
        gelu_grad_method: sigmoid
        activation_function: gelu
        add_bias: false""")
    head_dim_line = ("" if a.overcomplete == 1.0 else
                     f"\n        head_dim: {a.head_dim}"
                     "                       # REQUIRES head_dim decoupling patch")
    return f"""# {a.name} — single-block recurrent all-Boltzmann-MoE energy model.
# {a.note}
#
# ALL parameters live in ONE energy block, applied {a.n_iter}x. No GPT prefix.
# total params {fmt(a.total_params())} | active/token {fmt(a.active_params())} """ \
f"""({a.total_params()/a.active_params():.1f}x sparsity)
# experts are {100*a.moe_params()/a.total_params():.1f}% of all parameters
# D_qk = num_heads*head_dim = {a.D_qk} = {a.overcomplete:.2f}x d
#
# NOT YET TRAINED — emitted by scripts/design_single_block_moe_20260912.py for
# FLOPs/throughput benchmarking only. Set datasets/optimizer before training.

model_args:
  model_class: AutoModelForCausalLM
  pretrained_config:
    model_type: energy
    num_iterations: 1
    num_pre_layers: 0
    num_post_layers: 0
    num_layers: 1
    layer_iterations: [{a.n_iter}]
    iter_dropout_range_per_block: [0]
    hidden_size: {a.d}
    initializer_range: 0.02
    layer_norm_epsilon: 1.0e-05
    normalization_function: rmsnorm
    position_embedding_type: rope
    rope_dim: 64
    init_method: normal
    tie_word_embeddings: true
    energy_proj_type: psd_anti
    energy_proj_rank: {a.proj_rank}
    energy_antisym_rank: {a.antisym_rank}
    energy_apply_rayleigh: true
    energy_full_per_token_grad: true
    bos_token_id: 100257
    eos_token_id: 100257
    pad_token_id: 100256
    vocab_size: {VOCAB}
    max_position_embeddings: 4096
    sequence_mixer_blocks:
      - sequence_mixer_type: energy_attention
        num_attention_heads: {a.num_heads}
        num_key_value_heads: {a.num_heads}{head_dim_line}
        add_bias: false
        attention_multiplier: {1.0 / a.head_dim ** 0.5:.6f}
    mlp_blocks:
{mlp}
"""


if __name__ == "__main__":
    main()
