"""List the hosted models this ADAPTION_API_KEY can use, and check that each wanted one makes a function call.
Usage: python src/adaption_check.py [model_id ...]"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import provider
from testbed import tool_schemas

c = provider._client("adaption")
ids = []
for m in c._c.models.list(type="hosted").data if hasattr(c._c.models.list(type="hosted"), "data") else c._c.models.list(type="hosted"):
    mid = getattr(m, "id", None) or str(m)
    ids.append(mid)
    print(mid, getattr(m, "display_name", "") or "")
want = sys.argv[1:]
for mid in want:
    try:
        r = c.create(model=mid, messages=[{"role": "system", "content": "You are an agent on a forum."},
                                          {"role": "user", "content": "Look at the newest posts."}],
                     tools=tool_schemas(True), max_tokens=2000)
        m = r.choices[0].message
        print(f"[{mid}] tool_calls={[t.function.name for t in (m.tool_calls or [])]} text={m.content[:80]!r} usage={vars(r.usage) if r.usage else None}")
    except Exception as e:
        print(f"[{mid}] ERROR {e!r}"[:400])
