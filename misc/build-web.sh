#!/usr/bin/env bash
# Build the self-contained static web bundle into dist/ (or a custom directory).
# Usage: ./misc/build-web.sh [--lto] [OUT_DIR]
# Run inside `nix develop .#ci` or any environment with cargo (wasm32-unknown-unknown)
# and wasm-bindgen 0.2.130 on PATH.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="release"
OUT=""

for arg in "$@"; do
  case "$arg" in
    --lto) PROFILE="wasm-release" ;;
    *) OUT="$arg" ;;
  esac
done
OUT="${OUT:-$ROOT/dist}"

cd "$ROOT"

echo "[1/4] Building permanent-wasm (WebGPU engine worker, profile=$PROFILE)..."
cargo build --locked --profile "$PROFILE" --target wasm32-unknown-unknown -p permanent-wasm

echo "[2/4] Building permanent-web (Dioxus UI, profile=$PROFILE)..."
cargo build --locked --profile "$PROFILE" --target wasm32-unknown-unknown -p permanent-web

echo "[3/4] Generating wasm-bindgen JS/Wasm packages..."
rm -rf "$OUT"
mkdir -p "$OUT/pkg" "$OUT/pkg-engine"

wasm-bindgen \
  --target web \
  --remove-name-section \
  --out-dir "$OUT/pkg-engine" \
  "target/wasm32-unknown-unknown/$PROFILE/permanent_wasm.wasm"

wasm-bindgen \
  --target web \
  --remove-name-section \
  --out-name permanent_web \
  --out-dir "$OUT/pkg" \
  "target/wasm32-unknown-unknown/$PROFILE/permanent-web.wasm"

if command -v wasm-opt >/dev/null 2>&1; then
  echo "[3b] Optimizing Wasm binaries with wasm-opt -O3..."
  wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int \
    "$OUT/pkg/permanent_web_bg.wasm" -o "$OUT/pkg/permanent_web_bg.wasm"
  wasm-opt -O3 --enable-bulk-memory --enable-nontrapping-float-to-int \
    "$OUT/pkg-engine/permanent_wasm_bg.wasm" -o "$OUT/pkg-engine/permanent_wasm_bg.wasm"
fi

echo "[4/4] Copying static HTML and vendored assets..."
cp crates/permanent-web/index.html "$OUT/index.html"
cp -R crates/permanent-web/assets "$OUT/assets"

echo "Static bundle ready in $OUT"
ls -lh "$OUT/pkg/permanent_web_bg.wasm" "$OUT/pkg-engine/permanent_wasm_bg.wasm"
