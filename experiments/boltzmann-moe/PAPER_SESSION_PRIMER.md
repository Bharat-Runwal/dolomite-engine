# Primer for a dedicated ICLR-paper session

Read this, then `HANDOFF.md` §12, then start. Deliberately short: numbers live in ONE place and
this file points at it, because a second copy is a second thing to forget to update.

## Scope

Paper: `/u/ndehmamy/Code/overleaf/boltzmann-moe-ICLR-2026/` (Overleaf remote, branch `main`).
`main.tex` + `sec/{intro,theory,experiments,appendix}.tex`. Deadline is days away, so the job is
**correct + defensible**, not expanded.

Do NOT open `~/Code/energy/energy-GPT-neurips2026/` or the talk repo. They are archives on
superseded metrics (`avg9`/`avg10`) and reading them has already caused one wrong restatement.

## Metric, in one line

Everything is **Avg11** (11-task unweighted mean; MMLU and GSM8K-CoT as SEPARATE columns, never
averaged in). `experiments/eval_scripts/compute_avg11.py <run_dir>` is the only source. Avg11
runs ~3pp below the old avg10 because race and lambada sit near chance at our scale — that is a
scoring convention, not a model effect, so never compare an Avg11 against an avg10.

For language-modelling comparisons report **bits/byte**, not `word_perplexity`. They are
`word_ppl = exp(bpb * 3.7066)`, so ppl exponentially amplifies: one arm's regression reads 1.55x
in bpb and 7.8x in ppl. The ppl framing has already overstated a result in this project.

## Numbers: where they are and which are safe

`HANDOFF.md` §12.2 and §12.3 carry the current Avg11 / bits-per-byte tables with provenance.
Two standing traps:

* **`iclr_big_hop_pure_sink` is not comparable** — 4 GPUs against its counterpart's 8, so half
  the tokens. Its -0.59pp is undertraining. Strike it wherever it appears.
* **Anything about recurrence depth measured before 2026-09-16 is inverted.** The mu-at-eval bug
  made deeper look worse; fixed, deeper is better and T12 is the best pure arm. §12.1-12.2.

## Open `\CC` items

Listed in §12.7 (3). At the time of writing: the un-remeasured expert-cosine bound; the
token-scaling paragraph, pending the 90k arms in §12.6; and the 400M pure router comparison,
whose "+0.19pp in favour of energy" was measured with the SIGN-INVERTED router and must not be
quoted as support for energy routing.

## Coordination — the part that actually needs a rule

* **Only ONE session pushes to the Overleaf remote.** If you are the paper session, that is you.
  Always `git pull --rebase` before writing: the user edits in the Overleaf web UI, so incoming
  commits are theirs and are never a revert of yours. Diff an incoming commit against **its own
  parent**, not against your HEAD — doing the latter once made a legitimate edit look like a
  revert, and a force-push would have destroyed it.
* Pushing requires the user's confirmation each time.
* The engineering session (sparsity, runs, evals) can be reached with
  `SendMessage` — use `ListAgents` to find it. Ask it for a number rather than re-deriving one;
  it has the eval pipeline loaded.
* **Never invent or relabel an eval number.** If a cell has no measurement, it gets `\CC`.
