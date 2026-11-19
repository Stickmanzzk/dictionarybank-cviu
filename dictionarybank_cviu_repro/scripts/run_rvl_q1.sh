#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
CACHE="${CACHE:-$ROOT_DIR/cache/rvl_cdip/rvl400.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/rvl_cdip}"

mkdir -p "$OUTPUT_DIR/checkpoints" "$OUTPUT_DIR/eval" "$OUTPUT_DIR/logs"

for seed in 0 1 2; do
  "$PYTHON" "$ROOT_DIR/autoencoder/pacs_factor_sae.py" \
    --cache "$CACHE" \
    --target-cache "$CACHE" \
    --target-domain target \
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
    --out "$OUTPUT_DIR/checkpoints/seed${seed}.pt" \
    --report-json "$OUTPUT_DIR/checkpoints/seed${seed}.json" \
    > "$OUTPUT_DIR/logs/train_seed${seed}.log" 2>&1
done

"$PYTHON" "$ROOT_DIR/autoencoder/pacs_factor_sae.py" \
  --cache "$CACHE" \
  --target-cache "$CACHE" \
  --target-domain target \
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
  --out "$OUTPUT_DIR/checkpoints/wide192.pt" \
  --report-json "$OUTPUT_DIR/checkpoints/wide192.json" \
  > "$OUTPUT_DIR/logs/train_wide192.log" 2>&1

evaluate() {
  local name="$1"
  local budget="$2"
  shift 2
  "$PYTHON" "$ROOT_DIR/experiments/09_cviu_controls/evaluate_bank.py" \
    --cache "$CACHE" \
    --target-cache "$CACHE" \
    --checkpoints "$@" \
    --target-domain target \
    --active-budget "$budget" \
    --out "$OUTPUT_DIR/eval/${name}.pt" \
    --report-json "$OUTPUT_DIR/eval/${name}.json" \
    > "$OUTPUT_DIR/logs/eval_${name}.log" 2>&1
}

evaluate single_seed0 64 "$OUTPUT_DIR/checkpoints/seed0.pt"
evaluate single_seed1 64 "$OUTPUT_DIR/checkpoints/seed1.pt"
evaluate single_seed2 64 "$OUTPUT_DIR/checkpoints/seed2.pt"
evaluate bank_01 64 "$OUTPUT_DIR/checkpoints/seed0.pt" "$OUTPUT_DIR/checkpoints/seed1.pt"
evaluate bank_02 64 "$OUTPUT_DIR/checkpoints/seed0.pt" "$OUTPUT_DIR/checkpoints/seed2.pt"
evaluate bank_12 64 "$OUTPUT_DIR/checkpoints/seed1.pt" "$OUTPUT_DIR/checkpoints/seed2.pt"
evaluate bank_full 64 \
  "$OUTPUT_DIR/checkpoints/seed0.pt" \
  "$OUTPUT_DIR/checkpoints/seed1.pt" \
  "$OUTPUT_DIR/checkpoints/seed2.pt"
evaluate repeated_seed0_top192 192 \
  "$OUTPUT_DIR/checkpoints/seed0.pt" \
  "$OUTPUT_DIR/checkpoints/seed0.pt" \
  "$OUTPUT_DIR/checkpoints/seed0.pt"
evaluate wide192 192 "$OUTPUT_DIR/checkpoints/wide192.pt"

"$PYTHON" "$ROOT_DIR/experiments/10_rvl_q1_controls/analyze_rvl_q1.py" \
  --root "$OUTPUT_DIR" \
  > "$OUTPUT_DIR/logs/analyze_rvl.log" 2>&1

printf 'Results written to %s\n' "$OUTPUT_DIR"
