"""Smoke test: does each model make valid tool calls in the testbed? No judge."""
import sys, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent))
import provider
from testbed import Platform, load_profiles, run_episode

models = sys.argv[1].split(",")
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
B = int(sys.argv[3]) if len(sys.argv) > 3 else 10
P = Platform(5); profs = load_profiles()
out = Path(__file__).resolve().parent.parent / "runs" / "smoke.jsonl"; out.parent.mkdir(exist_ok=True)

def one(args):
    m, i = args
    t0 = time.time()
    try:
        ep = run_episode(provider.chat, m, P, profs[i], i, False, True, B, f"smoke:{m}:{i}")
    except Exception as e:
        return m, i, f"ERROR {e!r}"[:400]
    with open(out, "a", encoding="utf-8") as f: f.write(json.dumps(ep, default=str, ensure_ascii=False) + "\n")
    tools = [e["tool"] for e in ep["events"]]
    return m, i, f"{ep['stop_reason']} calls={ep['tool_calls']} turns={ep['llm_turns']} writes={len(ep['writes'])} esc={len(ep['escalations'])} seen_inj={ep['n_seen_injected']} {time.time()-t0:.0f}s tools={tools}"

with ThreadPoolExecutor(8) as ex:
    for r in ex.map(one, [(m, i) for m in models for i in range(n)]): print(*r, flush=True)
import collections
c = collections.defaultdict(lambda: [0, 0, 0.0])
for l in (open(provider.CALLS) if provider.CALLS.exists() else []):
    r = json.loads(l)
    if r["run_id"].startswith("smoke"):
        c[r["alias"]][0] += r["prompt_tokens"] or 0; c[r["alias"]][1] += r["completion_tokens"] or 0; c[r["alias"]][2] += r["cost_usd"]
for k, v in c.items(): print(k, "in", v[0], "out", v[1], f"${v[2]:.4f}")
print("total spend", provider.spent())
