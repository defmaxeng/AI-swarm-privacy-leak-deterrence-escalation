"""Judge episodes whose judging failed (judge_pending) and rewrite the run file in place.
Usage: python src/rejudge.py runs/<exp>.jsonl"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import judge_episode
from testbed import load_profiles

path = Path(sys.argv[1])
profiles = load_profiles()
eps = []
for l in open(path, encoding="utf-8"):
    try:
        eps.append(json.loads(l))
    except json.JSONDecodeError:
        print("skipping a damaged line")
todo = [e for e in eps if e.get("judge_pending")]
print(f"{len(todo)} of {len(eps)} episodes need judging")
for e in todo:
    try:
        judge_episode(e, profiles[e["persona"]], e["judge"])
        print("judged", e["key"], "leak", e["leak"], "leak_domain", e["leak_domain"])
    except Exception as ex:
        e["judge_error"] = repr(ex)[:500]
        print("still failing", e["key"], repr(ex)[:150])
tmp = path.with_suffix(".tmp")
with open(tmp, "w", encoding="utf-8") as f:
    for e in eps:
        f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
tmp.replace(path)
