#!/bin/bash

# If any command fails, exit immediately with that command's exit status
set -eo pipefail

rm -rf $(find docs/api_pydoctor -type l)
