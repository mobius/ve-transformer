#!/usr/bin/env bash
# Fetch exact image-framework commits without changing the LLM checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
source_dir=build/vendor/stable-diffusion.cpp
if [[ ! -d $source_dir/.git ]]; then
 git clone --no-checkout https://github.com/leejet/stable-diffusion.cpp.git "$source_dir"
 git -C "$source_dir" checkout --detach a1ded76da5818803fca97a3b433669ef727d32cf
 git -C "$source_dir" submodule update --init --depth 1 ggml
fi
[[ $(git -C "$source_dir" rev-parse HEAD) == a1ded76da5818803fca97a3b433669ef727d32cf ]]
[[ $(git -C "$source_dir/ggml" rev-parse HEAD) == 89c4413f5da6fb20cc796f16033d37f129be81fd ]]
[[ -z $(git -C "$source_dir" status --porcelain) ]]
printf 'Pinned SD framework and independent ggml verified\n'
