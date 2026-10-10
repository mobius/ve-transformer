#!/usr/bin/env bash
# Pin a compact SD-Turbo framework independently of the modern experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
dest=build/vendor/sd-turbo-baseline
revision=5eb15ef4d022bef4a391de4f5f6556e81fbb5024
ggml_revision=6fcbd60bc72ac3f7ad43f78c87e535f2e6206f58
if [[ ! -e $dest/.git ]]; then
 if git -C build/vendor/stable-diffusion.cpp cat-file -e "$revision^{commit}" 2>/dev/null; then
  git clone --shared --no-checkout build/vendor/stable-diffusion.cpp "$dest"
 else
  git clone --no-checkout --depth 1 https://github.com/leejet/stable-diffusion.cpp.git "$dest"
  git -C "$dest" -c fetch.recurseSubmodules=false fetch --depth 1 origin "$revision"
 fi
 git -C "$dest" checkout --detach "$revision"
 if git -C build/vendor/stable-diffusion.cpp/ggml cat-file -e "$ggml_revision^{commit}" 2>/dev/null; then
  git clone --shared --no-checkout build/vendor/stable-diffusion.cpp/ggml "$dest/ggml"
  git -C "$dest/ggml" checkout --detach "$ggml_revision"
 else
  git -C "$dest" submodule update --init --depth 1 ggml
 fi
fi
[[ $(git -C "$dest" rev-parse HEAD) == "$revision" ]]
[[ $(git -C "$dest/ggml" rev-parse HEAD) == "$ggml_revision" ]]
[[ -z $(git -C "$dest" status --porcelain) ]]
printf 'Pinned compact SD-Turbo framework verified\n'
