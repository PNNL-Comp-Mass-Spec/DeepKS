#!/bin/bash

# If any command fails, exit immediately with that command's exit status
set -eo pipefail

black -v --line-length 120 --preview **/*.py

# # Find all changed files for this commit
# # Compute the diff only once to save a small amount of time.
# CHANGED_FILES=$(git diff --name-only --cached --diff-filter=ACMR)
# # Get only changed files that match our file suffix pattern
# get_pattern_files() {
#     pattern=$(echo "$*" | sed "s/ /\$\\\|/g")
#     echo "$CHANGED_FILES" | { grep "$pattern$" || true; }
# }
# # Get all changed python files
# PY_FILES=$(get_pattern_files .py)
# echo $PY_FILES

# if [[ -n "$PY_FILES" ]]
# then
#     black -v --line-length 120 --experimental-string-processing **/*.py
# fi
