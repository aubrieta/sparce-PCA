#!/usr/bin/env bash
set -euo pipefail
SPCA_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$SPCA_ROOT"
export PYTHONPATH="$SPCA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
SPCA_PYTHON=${SPCA_PYTHON:-python3}
SPCA_OUT=${SPCA_OUT:-results/public-kernel-optimized}
SPCA_POLICY_ARGS=()
if [[ -n "${SPCA_ACCURACY_POLICY:-}" ]]; then
  SPCA_POLICY_ARGS=(--accuracy-policy "$SPCA_ACCURACY_POLICY")
fi
set +e
"$SPCA_PYTHON" -m sparse_pca.public_kernel run --out "$SPCA_OUT" --mpspdz /tmp/MP-SPDZ "${SPCA_POLICY_ARGS[@]}"
SPCA_STATUS=$?
set -e
"$SPCA_PYTHON" -m sparse_pca.public_kernel summarize --out "$SPCA_OUT"
"$SPCA_PYTHON" -m sparse_pca.public_kernel bundle --out "$SPCA_OUT"
exit "$SPCA_STATUS"
