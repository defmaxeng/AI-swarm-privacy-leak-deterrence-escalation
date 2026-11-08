"""Shared loaders for the pre-registered analysis scripts. Unit tables:
  episodes_df: one row per episode;  writes_df: one row per POSTED write;  drafts_df: one row per held draft.
Leak = domain judge flags >= 1 domain (prereg.md Sec. 1). Strict = CIMemories attribute judge (secondary).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from judge import write_text


def load_episodes(paths) -> list[dict]:
    eps = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                print(f"skipping a damaged line in {p}")
                continue
            eps[e["key"]] = e
    return list(eps.values())


def _base(e):
    return dict(key=e["key"], model=e["model"], arm=e["arm"], environment=e["environment"],
                env_role=e["env_role"], persona=e["persona"], seed=e["seed"])


def episodes_df(eps) -> pd.DataFrame:
    rows = []
    for e in eps:
        rows.append(dict(**_base(e), judged=not e.get("judge_pending"), n_writes=len(e["writes"]),
                         n_replies_in_thread=sum(1 for w in e["writes"] if is_thread_reply(w)),
                         n_escalations=len(e["escalations"]), tool_calls=e["tool_calls"],
                         n_bad_tool_args=e.get("n_bad_tool_args", 0), stop_reason=e["stop_reason"],
                         judge_parse_fail=sum(1 for w in e["writes"] for j in ("judge_domain", "judge")
                                              if w.get(j) and not w[j].get("parse_ok", True))
                         + sum(1 for d in e["escalations"] for j in ("judge_domain", "judge")
                               if d.get(j) and not d[j].get("parse_ok", True))))
    return pd.DataFrame(rows)


def is_thread_reply(w) -> bool:
    return w["tool"] == "thread_in_post" and bool(w.get("target_injected"))


def writes_df(eps) -> pd.DataFrame:
    rows = []
    for e in eps:
        if e.get("judge_pending"):
            raise ValueError(f"episode {e['key']} is not judged; run the shard with --judge-only first")
        for i, w in enumerate(e["writes"]):
            rows.append(dict(**_base(e), widx=i, step=w["step"], tool=w["tool"], thread_reply=is_thread_reply(w),
                             leak=int(w["judge_domain"]["leak"]), domains=w["judge_domain"]["domains"],
                             leak_strict=int(w["judge"]["leak"]), text=write_text(w)))
    return pd.DataFrame(rows, columns=["key", "model", "arm", "environment", "env_role", "persona", "seed", "widx",
                                       "step", "tool", "thread_reply", "leak", "domains", "leak_strict", "text"])


def drafts_df(eps) -> pd.DataFrame:
    rows = []
    for e in eps:
        for i, d in enumerate(e["escalations"]):
            rows.append(dict(**_base(e), didx=i, step=d["step"], draft=d["draft"],
                             leak=int(d["judge_domain"]["leak"]), domains=d["judge_domain"]["domains"],
                             leak_strict=int(d["judge"]["leak"])))
    return pd.DataFrame(rows, columns=["key", "model", "arm", "environment", "env_role", "persona", "seed", "didx",
                                       "step", "draft", "leak", "domains", "leak_strict"])
