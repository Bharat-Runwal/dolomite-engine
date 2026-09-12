"""Generate the Hopfield-MoE revival A/B ladder from the measured-dead baseline.

Baseline: math_fet_boltz_hopfield_rep_8gpt_1egpt6x_d1536_int8k_K8_lra32_itd3_lr1p5e3_33b_16gpu
  measured FF share of grad_E = 0.04%, effective_n_experts = 7.999/8,
  deleting the whole MoE branch moved ppl by +0.0003 over 87,997 tokens.

Three defects, three independent knobs, so the ladder attributes each one:

  g  hopfield_grad_scale  mean -> sqrt_consistent   branch magnitude   32x
  r  routing_norm         none -> zscore (+ tau)    routing sharpness  eff_n 8.0 -> ~2
  c  repulsion_form       signed -> abs             stop rewarding anti-alignment
  k  top_k                None -> 2                 sparsity (4x arithmetic)

REVERT: every knob defaults to the pre-2026-09-12 behaviour, so deleting the added
lines from a config reproduces the baseline exactly. `hopfield_grad_scale: mean` is
the specific one to restore if a run diverges — run 1714840 went NaN at step 3710
with the SUM form, and these settings sit 32x below it (see
_hopfield_grad_prefactor in energy_ff.py).

Probes run 4500 steps -- PAST STEP 3710, where run 1714840 diverged. A 1200-step
smoke could not have answered the stability question at all. `ga` is derived from
--ngpu to hold the baseline's 1.05M tokens/step (16 GPU x ga4 == 8 GPU x ga8), so
loss curves stay comparable across GPU counts. Default --ngpu 8 because a 2-node
16-GPU ask sat PEND behind ~1000 queued jobs; at 8 GPUs a probe is ~7.3 h and
reaches step 3710 at ~6.0 h. Checkpoints every 1000 steps.

  python experiments/boltzmann-moe/scripts/make_hopfield_revival_configs_20260912.py
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / ("configs/multi_block_ablation/math_fet_boltz_hopfield_rep_8gpt_1egpt6x"
               "_d1536_int8k_K8_lra32_itd3_lr1p5e3_33b_16gpu.yml")
OUT = REPO / "configs/hopfield_revival"
RESULTS = "/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/hopfield-revival"

# name suffix -> (mlp knob lines, human note)
VARIANTS = {
    # NOTE every rung below 'grc' pins repulsion_form: signed EXPLICITLY. The
    # library default changed to "squared" on 2026-09-12, so omitting it would
    # silently change two knobs at once and destroy the attribution.
    "g": (
        {"hopfield_grad_scale": "sqrt_consistent", "repulsion_form": "signed"},
        "grad prefactor 4/I_e -> 4/sqrt(I_e) (32x). Does the branch become "
        "load-bearing at all? Routing left flat on purpose, to isolate magnitude.",
    ),
    "gr": (
        {"hopfield_grad_scale": "sqrt_consistent", "routing_norm": "zscore",
         "temperature": 0.35, "repulsion_form": "signed"},
        "+ scale-free routing. zscore normalises to unit std across experts, so "
        "tau=0.35 is what takes effective_n_experts from ~5.6 to ~2.",
    ),
    "grc": (
        {"hopfield_grad_scale": "sqrt_consistent", "routing_norm": "zscore",
         "temperature": 0.35, "repulsion_form": "abs"},
        "+ repulsion fixed. 'abs' not 'squared' because it preserves the lambda=0.01 "
        "calibration from the signed-form sweeps (squared is ~3x weaker at the same "
        "lambda). Both are minimised at orthogonality.",
    ),
    "grck2": (
        {"hopfield_grad_scale": "sqrt_consistent", "routing_norm": "zscore",
         "temperature": 0.35, "repulsion_form": "abs", "top_k": 2},
        "+ sparse top-2 of 8. TRAINED sparse, not truncated post-hoc: h1_boltz_topk2 "
        "pays 0.0000 ppl for top-2 while the soft-trained h1 pays +0.68.",
    ),
    "ctl": (
        {"repulsion_form": "signed"},
        "CONTROL: baseline behaviour (repulsion_form pinned to the legacy 'signed' "
        "because the library default changed to 'squared'). Gives the reference loss "
        "curve and confirms the harness reproduces the dead branch.",
    ),
}


def build(tag: str, knobs: dict, note: str, steps: int, ngpu: int, ga: int) -> str:
    text = BASE.read_text()
    name = f"hopfield_revival_{tag}_probe{steps//100}" if steps <= 8000 else \
           f"hopfield_revival_{tag}_33b"

    # --- patch the energy-FF mlp block ---
    # A knob that ALREADY exists in the base config must be REPLACED, not appended:
    # a duplicate YAML key silently takes the last value, so inserting
    # `temperature: 0.35` above the base's `temperature: 1.0` would be a no-op.
    insert = {}
    for k, v in knobs.items():
        pat = re.compile(rf"^(\s+){k}: \S+.*$", re.M)
        if pat.search(text):
            text = pat.sub(lambda m, k=k, v=v: f"{m.group(1)}{k}: {v}"
                           f"   # CHANGED from baseline", text, count=1)
        else:
            insert[k] = v
    added = "".join(f"\n        {k}: {v}" for k, v in insert.items())
    text = text.replace(
        "        expert_kind: hopfield                            # single-W Hopfield experts",
        "        expert_kind: hopfield" + added, 1)
    assert "expert_kind: hopfield" in text
    # no duplicate keys inside the mlp block
    blk = text[text.index("mlp_type: EnergyFF_BoltzmannMoE"):text.index("tuning_args:")]
    keys = re.findall(r"^\s{8}(\w+):", blk, re.M)
    assert len(keys) == len(set(keys)), f"duplicate mlp keys in {name}: {keys}"

    # stale baseline comments that no longer describe this file
    text = text.replace("          # 32k × 1.05M tok/step = 33.5B (matches FET-1.5e-3)", "")
    text = text.replace("     # 16 GPU × mb=4 × ga=4 = 1.05M tok/step",
                        f"     # {ngpu} GPU x mb=4 x ga={ga} = 1.05M tok/step")

    # --- steps / schedule / parallelism ---
    text = re.sub(r"num_training_steps: \d+", f"num_training_steps: {steps}", text)
    text = re.sub(r"num_warmup_steps: \d+",
                  f"num_warmup_steps: {max(50, steps // 6)}", text)
    text = re.sub(r"num_decay_steps: \d+", f"num_decay_steps: {steps}", text)
    # Frequent checkpoints: the prefactor change carries divergence risk (run
    # 1714840 NaN'd at step 3710), so a blow-up should cost <=1000 steps of work.
    text = re.sub(r"save_interval: \d+", f"save_interval: {min(1000, max(200, steps // 2))}", text)
    text = re.sub(r"eval_interval: \d+", f"eval_interval: {steps}", text)
    text = re.sub(r"gradient_accumulation_steps: \d+",
                  f"gradient_accumulation_steps: {ga}", text)

    # --- names / paths ---
    text = re.sub(r"save_path: .*", f"save_path: {RESULTS}/{name}", text)
    text = re.sub(r"    name: .*", f"    name: {name}", text)

    header = (
        f"# {name}\n#\n"
        f"# {note}\n#\n"
        f"# Derived from {BASE.name} by\n"
        f"# scripts/make_hopfield_revival_configs_20260912.py. Baseline measured DEAD:\n"
        f"# FF share of grad_E = 0.04%, effective_n_experts = 7.999/8, deleting the\n"
        f"# whole MoE branch moved ppl +0.0003 over 87,997 tokens.\n#\n"
        f"# KNOBS SET HERE: {knobs if knobs else '(none - control)'}\n"
        f"# TO REVERT: delete those lines. Each defaults to pre-2026-09-12 behaviour.\n"
        f"# IF THIS DIVERGES: hopfield_grad_scale: mean restores the old prefactor\n"
        f"#   (run 1714840 NaN'd at step 3710 with the SUM form; these sit 32x below).\n"
        f"# {ngpu} GPUs x mb=4 x ga={ga} = 1.05M tok/step, iso with the 16-GPU baseline.\n#\n"
    )
    body = text[text.index("datasets:"):]
    return header + body


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4500)
    ap.add_argument("--ngpu", type=int, default=8,
                    help="GPUs per job. 8 = single node, which schedules; a 2-node "
                         "16-GPU ask sat PEND behind ~1000 queued jobs.")
    ap.add_argument("--tags", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    # hold tokens/step at the baseline 1.05M: 16 GPU x mb4 x ga4 == 8 GPU x mb4 x ga8
    ga = 4 * (16 // args.ngpu)
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for tag in args.tags:
        knobs, note = VARIANTS[tag]
        p = OUT / f"hopfield_revival_{tag}_probe{args.steps//100}.yml"
        p.write_text(build(tag, knobs, note, steps=args.steps, ngpu=args.ngpu, ga=ga))
        written.append(p)
    # the two most likely full runs, ready to go if the smokes pass
    for tag in ("grc", "grck2"):
        knobs, note = VARIANTS[tag]
        p = OUT / f"hopfield_revival_{tag}_33b.yml"
        p.write_text(build(tag, knobs, note, steps=32000, ngpu=16, ga=4))
        written.append(p)
    for p in written:
        print(f"  wrote {p.relative_to(REPO)}")
    print(f"\n  {len(written)} configs. Probes: {args.steps} steps / {args.ngpu} GPU "
          f"ga={ga} (past the step-3710 divergence point). Full: 32000 / 16 GPU ga=4.")


if __name__ == "__main__":
    main()
