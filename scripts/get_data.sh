#!/usr/bin/env bash
# Fetch the third-party inputs that are not committed (see .gitignore).
#   - Moltbook snapshot used by Priyanshu et al. (ref [35]), MIT license
#   - CIMemories profiles and prompts (Mireshghallah et al.), CC BY-NC 4.0
#   - The paper we build on (arXiv 2605.27766)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/moltbook data/raw/cimemories ext paper
for f in moltbook_posts.csv moltbook_submolts.csv README.md; do
  curl -sL -o "data/raw/moltbook/$f" "https://huggingface.co/datasets/ronantakizawa/moltbook/resolve/main/$f"
done
for f in cimemories_raw.json prompts_raw.yaml README.md; do
  curl -sL -o "data/raw/cimemories/$f" "https://huggingface.co/datasets/facebook/CIMemories/resolve/main/$f"
done
[ -d ext/CIMemories ] || git clone -q --depth 1 https://github.com/facebookresearch/CIMemories.git ext/CIMemories
curl -sL -o paper/paper.pdf https://arxiv.org/pdf/2605.27766v1
echo "done"
