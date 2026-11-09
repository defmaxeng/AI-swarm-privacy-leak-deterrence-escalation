"""Prepare a --part 1/2 shard folder for a model whose Phase 2 shard already ran partly (deviation D4).

  python scripts/rebalance_phase2.py glm-5.3-flash

Cells are split by index into part 0 (stays in runs/phase2/shard_<m>, original key) and part 1
(new folder runs/phase2/shard_<m>_p1, second key). For part-1 cells this copies, from the original folder into the new
one: finished judged episodes (so they are not rerun; merge_shards de-duplicates the identical copies), finished but
unjudged episodes (raw.jsonl, so they are judged, not rerun), and checkpoints of in-progress or crashed episodes are
MOVED (so they resume with the same seed). Run only while no shard process for the model is running.
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import prereg_params as pp  # noqa: E402

m = sys.argv[1]
P = pp.load(verbose=False)
sel = json.loads((ROOT / "results/phase1_selection.json").read_text(encoding="utf-8"))
T = pp.phase2_trials(P, sel["control_env"], sel["disclosure_env"], [m])
part1 = {pp.cell_key(t) for i, t in enumerate(T) if i % 2 == 1}
src, dst = ROOT / f"runs/phase2/shard_{m}", ROOT / f"runs/phase2/shard_{m}_p1"
(dst / "ckpt").mkdir(parents=True, exist_ok=True)
n = dict(episodes=0, raw=0, ckpt=0)
for name in ("episodes.jsonl", "raw.jsonl"):
    out = []
    if (src / name).exists():
        for line in open(src / name, encoding="utf-8"):
            try:
                if json.loads(line)["key"] in part1:
                    out.append(line if line.endswith("\n") else line + "\n")
            except json.JSONDecodeError:
                pass
    with open(dst / name, "a", encoding="utf-8") as f:
        f.writelines(out)
    n["episodes" if name == "episodes.jsonl" else "raw"] = len(out)
fname = {re.sub(r"[^A-Za-z0-9_.-]", "_", k) + ".json": k for k in part1}
for ck in list((src / "ckpt").glob("*.json")):
    if ck.name in fname:
        shutil.move(str(ck), str(dst / "ckpt" / ck.name))
        n["ckpt"] += 1
print(f"{m}: part 1 = {len(part1)} cells; copied {n['episodes']} finished, {n['raw']} raw; moved {n['ckpt']} checkpoints")
