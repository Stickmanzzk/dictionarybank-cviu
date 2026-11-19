#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/cache/domainnet}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/domainnet}"
FOLD_DIR="$CACHE_DIR/folds"

mkdir -p "$FOLD_DIR" "$OUTPUT_DIR/eval" "$OUTPUT_DIR/logs"

for target in 0 1 2 3 4 5; do
  "$PYTHON" "$ROOT_DIR/autoencoder/domainnet_fold_cache.py" \
    --train-cache "$CACHE_DIR/train.pt" \
    --test-cache "$CACHE_DIR/test.pt" \
    --target-domain "$target" \
    --out-dir "$FOLD_DIR"

  target_dir="$OUTPUT_DIR/target${target}"
  mkdir -p "$target_dir"

  for seed in 0 1 2; do
    "$PYTHON" "$ROOT_DIR/autoencoder/pacs_factor_sae.py" \
      --cache "$FOLD_DIR/target${target}_train.pt" \
      --target-cache "$FOLD_DIR/target${target}_test.pt" \
      --target-domain "$target" \
      --streams clip siglip dino \
      --content-latents 4096 \
      --content-k 64 \
      --nuisance-k 0 \
      --epochs 20 \
      --batch-size 512 \
      --class-weight 2 \
      --classifier-kind cosine \
      --domain-weight 0 \
      --adversarial-weight 0 \
      --alignment-weight 0.1 \
      --contrastive-weight 0.1 \
      --seed "$seed" \
      --out "$target_dir/seed${seed}.pt" \
      --report-json "$target_dir/seed${seed}.json" \
      > "$OUTPUT_DIR/logs/target${target}_seed${seed}.log" 2>&1

    "$PYTHON" "$ROOT_DIR/experiments/09_cviu_controls/evaluate_bank.py" \
      --cache "$FOLD_DIR/target${target}_train.pt" \
      --target-cache "$FOLD_DIR/target${target}_test.pt" \
      --checkpoints "$target_dir/seed${seed}.pt" \
      --target-domain "$target" \
      --active-budget 64 \
      --out "$OUTPUT_DIR/eval/target${target}_single_seed${seed}.pt" \
      --report-json "$OUTPUT_DIR/eval/target${target}_single_seed${seed}.json" \
      > "$OUTPUT_DIR/logs/target${target}_single_seed${seed}.log" 2>&1
  done

  "$PYTHON" "$ROOT_DIR/autoencoder/pacs_factor_sae.py" \
    --cache "$FOLD_DIR/target${target}_train.pt" \
    --target-cache "$FOLD_DIR/target${target}_test.pt" \
    --target-domain "$target" \
    --streams clip siglip dino \
    --content-latents 12288 \
    --content-k 192 \
    --nuisance-latents 256 \
    --nuisance-k 16 \
    --epochs 20 \
    --batch-size 512 \
    --class-weight 2 \
    --classifier-kind cosine \
    --domain-weight 0.2 \
    --adversarial-weight 0.1 \
    --alignment-weight 0.1 \
    --contrastive-weight 0.1 \
    --seed 0 \
    --out "$target_dir/wide192.pt" \
    --report-json "$target_dir/wide192.json" \
    > "$OUTPUT_DIR/logs/target${target}_wide192.log" 2>&1

  "$PYTHON" "$ROOT_DIR/experiments/09_cviu_controls/evaluate_bank.py" \
    --cache "$FOLD_DIR/target${target}_train.pt" \
    --target-cache "$FOLD_DIR/target${target}_test.pt" \
    --checkpoints "$target_dir/wide192.pt" \
    --target-domain "$target" \
    --active-budget 64 \
    --out "$OUTPUT_DIR/eval/target${target}_wide_k64.pt" \
    --report-json "$OUTPUT_DIR/eval/target${target}_wide_k64.json" \
    > "$OUTPUT_DIR/logs/target${target}_wide_k64.log" 2>&1

  for pair in "0 1" "0 2" "1 2"; do
    set -- $pair
    left="$1"
    right="$2"
    "$PYTHON" "$ROOT_DIR/experiments/09_cviu_controls/evaluate_bank.py" \
      --cache "$FOLD_DIR/target${target}_train.pt" \
      --target-cache "$FOLD_DIR/target${target}_test.pt" \
      --checkpoints "$target_dir/seed${left}.pt" "$target_dir/seed${right}.pt" \
      --target-domain "$target" \
      --active-budget 64 \
      --out "$OUTPUT_DIR/eval/target${target}_bank_${left}${right}.pt" \
      --report-json "$OUTPUT_DIR/eval/target${target}_bank_${left}${right}.json" \
      > "$OUTPUT_DIR/logs/target${target}_bank_${left}${right}.log" 2>&1
  done

  "$PYTHON" "$ROOT_DIR/experiments/09_cviu_controls/evaluate_bank.py" \
    --cache "$FOLD_DIR/target${target}_train.pt" \
    --target-cache "$FOLD_DIR/target${target}_test.pt" \
    --checkpoints \
      "$target_dir/seed0.pt" \
      "$target_dir/seed1.pt" \
      "$target_dir/seed2.pt" \
    --target-domain "$target" \
    --active-budget 64 \
    --out "$OUTPUT_DIR/eval/target${target}_bank_full.pt" \
    --report-json "$OUTPUT_DIR/eval/target${target}_bank_full.json" \
    > "$OUTPUT_DIR/logs/target${target}_bank_full.log" 2>&1

  "$PYTHON" "$ROOT_DIR/experiments/11_reviewer_controls/evaluate_raw.py" \
    --cache "$FOLD_DIR/target${target}_train.pt" \
    --target-cache "$FOLD_DIR/target${target}_test.pt" \
    --target-domain "$target" \
    --out "$OUTPUT_DIR/eval/target${target}_raw_multistream.pt" \
    --report-json "$OUTPUT_DIR/eval/target${target}_raw_multistream.json" \
    > "$OUTPUT_DIR/logs/target${target}_raw.log" 2>&1
done

for target in 0 1 2 3 4 5; do
  "$PYTHON" "$ROOT_DIR/experiments/11_reviewer_controls/diagnose_bank.py" \
    --target-cache "$FOLD_DIR/target${target}_test.pt" \
    --target-domain "$target" \
    --checkpoints \
      "$OUTPUT_DIR/target${target}/seed0.pt" \
      "$OUTPUT_DIR/target${target}/seed1.pt" \
      "$OUTPUT_DIR/target${target}/seed2.pt" \
    --prediction "$OUTPUT_DIR/eval/target${target}_bank_full.pt" \
    --out "$OUTPUT_DIR/support_target${target}.json" \
    > "$OUTPUT_DIR/logs/target${target}_support.log" 2>&1
done

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
support = {
    target: json.loads(
        (root / f"support_target{target}.json").read_text(encoding="utf-8")
    )
    for target in range(6)
}
(root / "support_diagnostics.json").write_text(
    json.dumps(support, indent=2),
    encoding="utf-8",
)
PY

"$PYTHON" "$ROOT_DIR/experiments/11_reviewer_controls/analyze_ensembles.py" \
  --root "$OUTPUT_DIR" \
  --bootstrap-samples 10000 \
  > "$OUTPUT_DIR/logs/analyze_ensembles.log" 2>&1

printf 'Results written to %s\n' "$OUTPUT_DIR"
