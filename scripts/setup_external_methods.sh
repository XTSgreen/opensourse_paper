#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/external_methods/sources"
declare -A repos=(
  [wot]=https://github.com/broadinstitute/wot.git
  [moscot]=https://github.com/theislab/moscot.git
  [lineageot]=https://github.com/aforr/LineageOT.git
  [tigon]=https://github.com/yutongo/TIGON.git
  [mioflow]=https://github.com/KrishnaswamyLab/MIOFlow.git
  [cellrank2]=https://github.com/scverse/cellrank.git
  [prescient]=https://github.com/gifford-lab/prescient.git
)
declare -A commits=(
  [wot]=ca5e94f05699997b01cf5ae13383f9810f0613f6
  [moscot]=440093ccbb8e70de209157d91da839c55b897821
  [lineageot]=6081b402074f7e5934e729e81669aef430219da8
  [tigon]=1ed92cfcc250415fc01b4d344a308b0680cc9635
  [mioflow]=36365403d0f23cc3ad1065781c7331bf81debf4e
  [cellrank2]=d7191d75684c86b58adbb317c8aae7d06f2682f3
  [prescient]=50971c7d495e8763eaa60f83af91f51555ed7ece
)
for name in "${!repos[@]}"; do
  destination="$root/external_methods/sources/$name"
  if [[ ! -d "$destination/.git" ]]; then git clone "${repos[$name]}" "$destination"; fi
  git -C "$destination" fetch --quiet --all --tags
  git -C "$destination" checkout --quiet --detach "${commits[$name]}"
  head="$(git -C "$destination" rev-parse HEAD)"
  [[ "$head" == "${commits[$name]}" ]] || { echo "Pinned commit verification failed for $name" >&2; exit 1; }
done
echo "Source repositories are checked out at the exact commits recorded in SOURCE_PINS.yaml. Create the four method environments from external_methods/environments/*.yaml."
if [[ "${1:-}" == "--create-envs" ]]; then
  command -v conda >/dev/null 2>&1 || { echo "Conda is required for --create-envs" >&2; exit 1; }
  for environment in "$root"/external_methods/environments/*.yaml; do
    conda env create --file "$environment"
  done
fi
