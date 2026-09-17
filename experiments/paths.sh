#!/bin/bash
# ============================================================================================
# paths.sh -- the ONE place machine-specific absolute paths are defined.
#
# Source it from any launcher under experiments/ instead of hardcoding paths:
#
#     . "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/experiments/paths.sh"
#     #  ^ adjust the ../ count to your script's depth; REPO_ROOT is then already correct
#
# It defines, and never overwrites, three variables:
#
#   REPO_ROOT   this checkout. DERIVED by self-location -- never configure it. Works from any
#               cwd and in any clone, so a colleague's clone needs no edit at all.
#   VENV        the Python venv to activate. Defaults to the shared one on this cluster;
#               override with  export DOLOMITE_VENV=/path/to/your/.venv
#   DATA_ROOT   the pretraining-data / tokenizer root. Defaults to this cluster's;
#               override with  export DOLOMITE_DATA_ROOT=/path/to/data
#
# WHY: the absolute paths used to be copy-pasted into 226 shell scripts. Concentrating them
# here means a different machine needs two exports rather than a repo-wide sed, and the repo
# carries the site-specific strings in exactly one file.
#
# NOTE the scope limit: this does NOT cover configs/**/*.yml. Those paths (save_path,
# load_path, data_path, data_cache_path, tokenizer_name) are read by the trainer, are baked
# into every existing checkpoint's saved config, and are what a resume matches against.
# Rewriting them would break resumption of in-flight runs, so they are deliberately untouched.
# ============================================================================================

# REPO_ROOT: derived, not configured. paths.sh lives at <root>/experiments/paths.sh.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Site defaults. Override by exporting the DOLOMITE_* variable before calling a launcher.
VENV="${DOLOMITE_VENV:-/proj/dmfexp/nima/Code/nanoGPT-og/.venv}"
DATA_ROOT="${DOLOMITE_DATA_ROOT:-/proj/datasets}"

# Kept for the many scripts that still say REPO=...
REPO="$REPO_ROOT"

export REPO_ROOT REPO VENV DATA_ROOT
