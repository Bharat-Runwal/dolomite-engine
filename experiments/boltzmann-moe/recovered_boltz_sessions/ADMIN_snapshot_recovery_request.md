# Draft email — GPFS snapshot recovery request

**To:** <cluster storage / HPC admin>
**Subject:** Snapshot recovery request — 2 deleted files in my home fileset (GPFS ess6000-1)

Hi <admin name>,

Could you please help me recover two files that were deleted from my home
directory today (2026-09-07)? They were removed by an application-level cleanup,
not by me, and I do not have read access to the fileset snapshots to restore them
myself:

- `/gpfs/ess6000-1/home/user.home/ndehmamy/.snapshots` → *Permission denied*
- `/gpfs/ess6000-1/.snapshots` → root-only

**Filesystem:** GPFS / IBM Spectrum Scale, device **ess6000-1**
**Home fileset internal path:** `/gpfs/ess6000-1/home/user.home/ndehmamy`
(mounts as `/u/ndehmamy` on login/compute nodes)

**Files to restore** (they existed intact until earlier today; deleted ~2026-09-07):

1. `/u/ndehmamy/.claude/projects/-proj-dmfexp-nima-Code-GPT-experiments/07d946f4-3c69-445b-8177-3e08d77bcc65.jsonl`
   - last modified 2026-08-02, ~16 MB

2. `/u/ndehmamy/.claude/projects/-proj-dmfexp-nima-Code-mucus-layer-modeling/ebb28c30-b6b8-427e-ae85-73ea08efe110.jsonl`
   - last modified 2026-07-25, ~5 MB

Any snapshot taken **before 2026-09-07** (e.g. last night's) should still contain
both files, since they were only deleted today. If you could copy them to a scratch
location I can read — e.g. `/u/ndehmamy/restore/` — or restore them in place, that
would be perfect.

**Lower-priority / longer shot:** I'm also missing an older session file (a
"boltzmann-moe" / EGPT experiment log) that was likely deleted back in **May–June
2026**. If your snapshot or TSM/backup retention reaches that far, I'd be grateful
for anything under
`/u/ndehmamy/.claude/projects/*GPT-experiments*/` or
`/u/ndehmamy/.claude/projects/*dolomite-engine*/` with a modification date in
**April–May 2026**. I understand this may be past your retention window.

For context: these are Claude Code CLI session transcripts (plain JSONL). The tool
auto-purges transcripts older than 30 days by last-modified time; I've since raised
that retention limit so it won't recur.

Thanks very much,
Nima Dehmamy
