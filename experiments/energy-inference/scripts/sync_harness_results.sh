#!/usr/bin/env bash
# sync_harness_results.sh
#
# Stages all small `harness_*.json` files under
# experiments/energy-inference/results/ for commit, updating .gitignore as needed.
#
# Why: the .gitignore on `nima-energy-inference` excludes everything under
# `experiments/energy-inference/results/` except *.pdf. The harness_*.json files
# (regular eval + BBH eval) live in <model>/unsharded/ alongside multi-GB model
# weights and `global_step*/` checkpoint dirs. We want to commit the JSONs (small)
# while keeping the weights and checkpoint state ignored.
#
# Safety: rejects any candidate file larger than MAX_PER_FILE_MB (default 10 MB)
# so a stray huge file can never sneak in. Use --dry-run to preview without
# touching anything.
#
# Usage (from dolomite-engine repo root, on branch nima-energy-inference):
#   bash sync_harness_results.sh             # update gitignore + stage
#   bash sync_harness_results.sh --dry-run   # report only, no changes

set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" ]]; then
    echo "ERROR: not inside a git repo. cd to dolomite-engine first." >&2
    exit 1
fi
cd "${REPO_ROOT}"

RESULTS_DIR="experiments/energy-inference/results"
EXCEPTION_LINE='!experiments/energy-inference/results/**/harness_*.json'
MAX_PER_FILE_MB="${MAX_PER_FILE_MB:-10}"
MAX_PER_FILE_BYTES=$(( MAX_PER_FILE_MB * 1024 * 1024 ))

if [[ ! -d "${RESULTS_DIR}" ]]; then
    echo "ERROR: ${RESULTS_DIR} not found. Are you on branch nima-energy-inference?" >&2
    exit 1
fi

# 1. Discover all harness_*.json files (covers harness_results_* and harness_bbh_results_*)
mapfile -t ALL_FILES < <(find "${RESULTS_DIR}" -type f -name 'harness_*.json' | sort)

if [[ ${#ALL_FILES[@]} -eq 0 ]]; then
    echo ">> No harness_*.json files found under ${RESULTS_DIR}/"
    echo "   Did you run from the right server/branch? Exiting."
    exit 0
fi

# 2. Size sanity check: reject any oversized file (defense against accidental matches)
SAFE_FILES=()
REJECTED_FILES=()
for f in "${ALL_FILES[@]}"; do
    size_bytes=$(stat -c '%s' "$f" 2>/dev/null || stat -f '%z' "$f")
    if (( size_bytes > MAX_PER_FILE_BYTES )); then
        REJECTED_FILES+=("$f ($(numfmt --to=iec --suffix=B "$size_bytes" 2>/dev/null || echo "${size_bytes}B"))")
    else
        SAFE_FILES+=("$f")
    fi
done

if [[ ${#REJECTED_FILES[@]} -gt 0 ]]; then
    echo "WARNING: ${#REJECTED_FILES[@]} file(s) exceed ${MAX_PER_FILE_MB} MB — NOT staging:"
    for r in "${REJECTED_FILES[@]}"; do echo "  ! $r"; done
    echo "   (raise the cap with MAX_PER_FILE_MB=N if these are legitimate)"
    echo
fi

if [[ ${#SAFE_FILES[@]} -eq 0 ]]; then
    echo ">> No files passed the size guard. Exiting without changes."
    exit 0
fi

echo ">> ${#SAFE_FILES[@]} harness_*.json files pass size guard (<= ${MAX_PER_FILE_MB} MB each)"
echo

# 3. Per-model breakdown
echo "Per-model file counts:"
for f in "${SAFE_FILES[@]}"; do
    echo "$(dirname "$(dirname "$f")")"
done | sort | uniq -c | sort -k2 | awk '{printf "  %4d  %s\n", $1, $2}'
echo

# 4. Pattern breakdown (regular vs BBH vs other)
echo "By filename pattern:"
for f in "${SAFE_FILES[@]}"; do
    bn=$(basename "$f")
    case "$bn" in
        harness_results_*)      echo "harness_results_*.json" ;;
        harness_bbh_results_*)  echo "harness_bbh_results_*.json" ;;
        *)                      echo "$bn" ;;
    esac
done | sort | uniq -c | awk '{printf "  %4d  %s\n", $1, $2}'
echo

# 5. Total size
TOTAL_SIZE=$(du -ch "${SAFE_FILES[@]}" | tail -1 | awk '{print $1}')
echo ">> Total size to stage: ${TOTAL_SIZE}"
echo

if (( DRY_RUN )); then
    echo "(--dry-run: not modifying .gitignore or staging anything)"
    exit 0
fi

# 6. Update .gitignore with whitelist exception
if ! grep -qF "${EXCEPTION_LINE}" .gitignore; then
    echo ">> Adding ${EXCEPTION_LINE} to .gitignore"
    awk -v line="${EXCEPTION_LINE}" '
        { print }
        /^!experiments\/energy-inference\/results\/.*\*\.pdf$/ && !done {
            print line
            done = 1
        }
    ' .gitignore > .gitignore.tmp
    mv .gitignore.tmp .gitignore
else
    echo ">> .gitignore already whitelists harness_*.json"
fi

# 7. Belt-and-suspenders: confirm nothing scary is about to be staged.
#    `git check-ignore` should report each file as NOT ignored after the gitignore update.
echo
echo ">> Verifying that only the intended files are unignored..."
LEAKED=()
while IFS= read -r unignored; do
    case "$(basename "$unignored")" in
        harness_*.json|*.pdf) ;;  # expected
        *) LEAKED+=("$unignored") ;;
    esac
done < <(git ls-files --others --exclude-standard -- "${RESULTS_DIR}")

if [[ ${#LEAKED[@]} -gt 0 ]]; then
    echo "ERROR: unexpected non-harness, non-PDF files would be unignored:" >&2
    for l in "${LEAKED[@]}"; do echo "  ! $l" >&2; done
    echo "Aborting before staging. Investigate .gitignore." >&2
    exit 1
fi

# 8. Stage .gitignore + the safe JSON files
echo ">> Staging .gitignore + ${#SAFE_FILES[@]} JSON files..."
git add .gitignore
# add in batches to avoid argv overflow on huge file lists
printf '%s\n' "${SAFE_FILES[@]}" | xargs -d '\n' git add --

# 9. Final confirmation: review what's staged
echo
echo "Staged change summary:"
git diff --cached --stat | tail -20
echo

# 10. Sanity check: total staged size
STAGED_BYTES=$(git diff --cached --numstat \
    | awk 'NR>0 { print $3 }' \
    | xargs -I{} stat -c '%s' "{}" 2>/dev/null \
    | awk '{ s+=$1 } END { print s+0 }')
echo ">> Total bytes staged (added files only): $(numfmt --to=iec --suffix=B "${STAGED_BYTES}" 2>/dev/null || echo "${STAGED_BYTES}B")"
if (( STAGED_BYTES > 200 * 1024 * 1024 )); then
    echo "WARNING: staged size exceeds 200 MB — review before committing!" >&2
fi
echo

echo "Done. Suggested next steps:"
echo
echo "  git commit -m \"Add harness eval JSON results for paper plots/tables\""
echo "  git push origin nima-energy-inference   # on the server, origin is your fork"
