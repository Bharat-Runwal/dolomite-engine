#!/bin/bash
# =============================================================================================
# LAUNCHER FOR EVERY ICLR-26 PAPER ARM. Written 2026-09-21.
#
#   bash configs/iclr_26/launch/launch_paper_arms.sh            # print the plan, submit NOTHING
#   bash configs/iclr_26/launch/launch_paper_arms.sh <arm> ...  # submit those arms
#   bash configs/iclr_26/launch/launch_paper_arms.sh ALL_400M   # every 400M arm
#   QUEUE=preemptable bash ... <arm>                            # default is ebm (normal/grp_ebm)
#
# It is a thin wrapper over experiments/boltzmann-moe/scripts/bsub/submit_train.sh, which encodes
# the node shape, the ptile/blaunch rules, the ut<0.5 host filter, BAD_HOSTS, the launch ledger and
# run-time load_args resolution. NEVER hand-roll a training bsub -- see CLAUDE.md for the two
# failures that cost on 2026-09-16 alone.
#
# ---------------------------------------------------------------------------------------------
# GPU COUNT IS NOT IN THE CONFIG. It is the 3rd argument here, and it is what makes tokens/step
# correct: tokens/step = GPUS x micro_batch_size x gradient_accumulation_steps x sequence_length.
# Every arm below is pinned to the count that yields its intended budget. Changing the GPU count
# WITHOUT changing ga silently halves or doubles the token budget -- that happened three times in
# one session (HANDOFF 14.2b).
#   134M arms: 262,144 tok/step x 122,070 steps = 32.0B
#   400M arms: 524,288 tok/step x  61,035 steps = 32.0B
#
# WANDB__SERVICE_WAIT=300 is exported because wandb's 30 s default timed out on a busy host and
# killed a 400M arm at rank 0 (job 1810260).
#
# TRANSIENT MULTI-NODE FAILURES ARE EXPECTED, ~50% (HANDOFF 13.9 item 3). If an arm dies in <60 s
# with `gloo ... Connection closed by peer` or `lsb_launch(): Failed`, that is NOT your config --
# just resubmit. Only investigate if it fails the same way three times.
# =============================================================================================
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
SUB=$REPO/experiments/boltzmann-moe/scripts/bsub/submit_train.sh
QUEUE="${QUEUE:-ebm}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"

# name | config (relative to $REPO) | gpus | wall | mem | note
ARMS=(
# ---- 400M tier: THE PRIORITY. Both sides of the headline comparison live here. --------------
"cmix_400M_hybrid_sparse|configs/iclr_26/scaling/cmix_400M_hybrid_sparse.yml|8|24:00|160G|P0 energy hybrid 6G1x6E, bank 64.8% of non-emb"
"abl_B_400M_6G1x6S|configs/iclr_26/scaling/abl_B_400M_6G1x6S.yml|8|24:00|160G|P0 FLOP-matched Switch 6G1x6S -- the correct baseline"
"cmix_400M_sandwich_sparse|configs/iclr_26/scaling/cmix_400M_sandwich_sparse.yml|8|24:00|160G|sandwich 1G1x4E1G, bank 87.4% -- most EGPT-heavy arm"
"abl_H_400M_6G6E_deep|configs/iclr_26/scaling/abl_H_400M_6G6E_deep.yml|8|24:00|160G|deep energy, 6 DISTINCT blocks, no recurrence"
"abl_H_400M_6G6S_deep|configs/iclr_26/scaling/abl_H_400M_6G6S_deep.yml|8|24:00|160G|deep Switch twin of the above"
"abl_H_400M_6G6G_deep_isoactive|configs/iclr_26/scaling/abl_H_400M_6G6G_deep_isoactive.yml|8|24:00|160G|12G dense control, iso-ACTIVE to 6G6E (238M total, NOT 400M)"
"abl_G_400M_6G1x6S1x6S|configs/iclr_26/scaling/abl_G_400M_6G1x6S1x6S.yml|8|24:00|160G|two recurrent Switch blocks"
"abl_G_400M_6G1x6E1x6E_4gpu|configs/iclr_26/scaling/abl_G_400M_6G1x6E1x6E_4gpu.yml|4|24:00|160G|two recurrent ENERGY blocks. MUST be single-node (persist_mu wedges multi-node); 8.5 s/step so 32B needs ~126 h"
"cmix_400M_baseline_switch|configs/iclr_26/scaling/cmix_400M_baseline_switch.yml|8|24:00|160G|unmatched Switch 6G1S -- UNDER-PROVISIONED by 12.9% FLOPs, keep only as the historical baseline"
# ---- 1B ------------------------------------------------------------------------------------
"cmix1B_12L_gptDense_32B|configs/iclr_26/scaling/cmix1B_12L_gptDense_32B.yml|8|24:00|160G|colleagues' 12-distinct-layer architecture ported"
# ---- 134M tier: COMPLETE. Re-run only to add seeds. ----------------------------------------
"cmix_134M_hybrid_32B_sparse|configs/iclr_26/scaling/cmix_134M_hybrid_32B_sparse.yml|8|24:00|96G|134M energy hybrid (headline)"
"abl_B_134M_6G1x6S|configs/iclr_26/scaling/abl_B_134M_6G1x6S.yml|8|24:00|96G|134M FLOP-matched Switch"
"abl_E_134M_6G1x6E_baseEGPT|configs/iclr_26/scaling/abl_E_134M_6G1x6E_baseEGPT.yml|4|24:00|96G|energy block, NO MoE -- iso-active AND iso-FLOP with the hybrid"
"abl_F_134M_6G_dense_isoactive|configs/iclr_26/scaling/abl_F_134M_6G_dense_isoactive.yml|4|24:00|96G|GPT-only iso-ACTIVE"
"abl_D_134M_6G_dense_isototal|configs/iclr_26/scaling/abl_D_134M_6G_dense_isototal.yml|4|24:00|96G|GPT-only iso-TOTAL (+9% active -- not a like-for-like row)"
"cmix_134M_hyb_w1w2_sparse_surr_32B|configs/iclr_26/scaling/cmix_134M_hyb_w1w2_sparse_surr_32B.yml|4|24:00|96G|w1w2 sparse(surrogate)"
"cmix_134M_hyb_w1w2_surrMLP_32B|configs/iclr_26/scaling/cmix_134M_hyb_w1w2_surrMLP_32B.yml|4|24:00|96G|w1w2 dense, head as ROUTER"
"abl_I_134M_w1w2_sparse_surr_projUncon|configs/iclr_26/scaling/abl_I_134M_w1w2_sparse_surr_projUncon.yml|4|24:00|96G|proj A/B: unconstrained. LOSES to psd_anti -- do not adopt"
"cmix_134M_sandwich_32B_sparse|configs/iclr_26/scaling/cmix_134M_sandwich_32B_sparse.yml|8|24:00|96G|energy block moved (5G1x6E1G), NOT a sandwich"
"cmix_134M_pure_32B_sparse|configs/iclr_26/scaling/cmix_134M_pure_32B_sparse.yml|8|24:00|96G|pure recurrent 1x12E -- worst arm, keep as the negative"
"cmix_134M_gptswitch_32B|configs/iclr_26/scaling/cmix_134M_gptswitch_32B.yml|8|24:00|96G|unmatched Switch 6G1S at 134M"
)

want=("$@")
[ ${#want[@]} -eq 0 ] && { printf '%s\n' "PLAN ONLY -- nothing submitted. Pass arm names, or ALL_400M." ""; }
printf '%-38s %5s %-6s %s\n' ARM GPUS QUEUE NOTE
for row in "${ARMS[@]}"; do
  IFS='|' read -r nm cfg g wall mem note <<<"$row"
  printf '%-38s %5s %-6s %s\n' "$nm" "$g" "$QUEUE" "$note"
done
[ ${#want[@]} -eq 0 ] && exit 0

echo; echo "submitting..."
for row in "${ARMS[@]}"; do
  IFS='|' read -r nm cfg g wall mem note <<<"$row"
  hit=0
  for w in "${want[@]}"; do
    [ "$w" = "$nm" ] && hit=1
    [ "$w" = "ALL_400M" ] && [[ "$nm" == *400M* ]] && hit=1
  done
  [ "$hit" -eq 0 ] && continue
  # 8 GPUs default to 2 nodes x 4 -- the shape every 400M arm has actually run at. A single
  # host with 8 free GPUs is usually unobtainable here (565 hosts at their GPU limit, job 1798342
  # sat PEND 25 min). The 4-GPU arms are single-node by construction.
  gpn=""; [ "$g" = "8" ] && gpn="4"
  echo "--- $nm ($g GPUs, $QUEUE)"
  bash "$SUB" "$nm" "$cfg" "$g" "$QUEUE" "$wall" "$mem" $gpn
done
echo
echo "VERIFY IN 60 s -- an arm that dies fast is usually transient, not your config:"
echo "  bjobs -o 'jobid job_name stat'"
echo "  tail -3 \$HOME/bsub_logs/<name>_<jobid>.stderr      # step lines are in STDERR, not stdout"
echo "  grep -m1 DeviceMesh \$HOME/bsub_logs/<name>_<jobid>.stderr   # must NOT say 'cpu'"
