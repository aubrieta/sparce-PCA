#!/usr/bin/env bash
set -euo pipefail
SPCA_MPSPDZ=${1:-/tmp/MP-SPDZ}
SPCA_REVISION=f5f0cd2433c61d464666cbae19da16ac197d7a75
if [[ ! -d "$SPCA_MPSPDZ" ]]; then
  git clone https://github.com/data61/MP-SPDZ.git "$SPCA_MPSPDZ"
  git -C "$SPCA_MPSPDZ" checkout "$SPCA_REVISION"
fi
git -C "$SPCA_MPSPDZ" rev-parse HEAD
make -C "$SPCA_MPSPDZ" -j4 replicated-ring-party.x
