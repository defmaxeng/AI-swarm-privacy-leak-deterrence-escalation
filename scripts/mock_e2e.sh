#!/usr/bin/env bash
# Offline end-to-end run of the pre-registered pipeline with the mock agent and judge (no API calls).
# Usage: bash scripts/mock_e2e.sh [OUT_DIR] [N_BOOT]   (N_BOOT defaults to the prereg value)
set -euo pipefail
cd "$(dirname "$0")/.."
M="${1:-runs/mock_e2e}"
NB="${2:-}"
rm -rf "$M"
python src/run_prereg.py --phase smoke --out "$M/smoke" --mock | tail -1
python src/run_prereg.py --phase 1 --out "$M/phase1/shard_a" --models glm-5.3-flash qwen3.8-27b --mock | tail -1
python src/run_prereg.py --phase 1 --out "$M/phase1/shard_b" --models deepseek-v4-flash gemma-4-26b-a4b --mock | tail -1
python src/merge_shards.py --phase 1 --shards "$M"/phase1/shard_* --out "$M/phase1_merged.jsonl" | tail -1
python src/phase1_select.py "$M/phase1_merged.jsonl" --out "$M/phase1_selection.json" | tail -2
for arm_shard in a b; do
  if [ $arm_shard = a ]; then MS="glm-5.3-flash qwen3.8-27b"; else MS="deepseek-v4-flash gemma-4-26b-a4b"; fi
  python src/run_prereg.py --phase 2 --out "$M/phase2/shard_$arm_shard" --models $MS \
      --selection "$M/phase1_selection.json" --mock | tail -1
done
python src/merge_shards.py --phase 2 --shards "$M"/phase2/shard_* --out "$M/phase2_merged.jsonl" \
    --selection "$M/phase1_selection.json" | tail -1
python src/phase2_analyze.py "$M/phase2_merged.jsonl" --selection "$M/phase1_selection.json" \
    --out "$M/phase2_results" ${NB:+--n-boot $NB} > "$M/phase2_analyze.log"
echo "analysis: $M/phase2_results/readout.md"
python src/sample_for_labels.py "$M/phase2_merged.jsonl" --out "$M/labels" | tail -1
python src/second_judge.py "$M/phase2_merged.jsonl" --sample "$M/labels/second_judge_sample.csv" \
    --out "$M/labels/second_judge.csv" --mock | sed -n 2p
echo "mock end-to-end OK"
