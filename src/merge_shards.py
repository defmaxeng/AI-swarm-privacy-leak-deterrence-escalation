"""Merge shard folders into one episodes file, de-duplicated on (model, persona, environment, arm, seed),
and list every expected cell that is missing. Reports counts only (no leakage).

  python src/merge_shards.py --phase 1 --shards runs/phase1/* --out runs/phase1_merged.jsonl
  python src/merge_shards.py --phase 2 --shards runs/phase2/* --out runs/phase2_merged.jsonl

Duplicates: the judged copy wins over a pending one; among equals the last one read wins (and is reported).
Exit code 1 if any expected cell is missing or still pending judgment.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prereg_params as pp  # noqa: E402
from run_prereg import load_selection, read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def dedup_key(e: dict) -> tuple:
    return tuple(e[k] for k in pp.CELL_FIELDS)


def merge(episode_lists: list[list[dict]]) -> tuple[dict, list[tuple]]:
    best, dups = {}, []
    for eps in episode_lists:
        for e in eps:
            k = dedup_key(e)
            if k in best:
                dups.append(k)
                if best[k].get("judge_pending") is False and e.get("judge_pending"):
                    continue
            best[k] = e
    return best, dups


def expected_cells(P, phase, selection=None):
    if phase == "1":
        trials = pp.phase1_trials(P)
    elif phase == "2":
        ctrl, disc, _ = load_selection(Path(selection))
        trials = pp.phase2_trials(P, ctrl, disc)
    elif phase == "followup":
        _, disc, _ = load_selection(Path(selection))
        trials = pp.followup_trials(P, disc)
    else:
        trials = pp.smoke_trials(P)
    return {dedup_key(t): t for t in trials}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["smoke", "1", "2", "followup"])
    ap.add_argument("--shards", nargs="+", required=True, help="shard folders (each holds episodes.jsonl)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", default=str(ROOT / "results/phase1_selection.json"))
    a = ap.parse_args()
    P = pp.load()
    lists = []
    for s in a.shards:
        s = Path(s)
        eps = read_jsonl(s / "episodes.jsonl")
        raw_only = [e for e in read_jsonl(s / "raw.jsonl") if e["key"] not in {x["key"] for x in eps}]
        print(f"{s}: {len(eps)} judged episodes, {len(raw_only)} finished but not judged")
        lists.append(raw_only)  # read first so a judged copy of the same cell replaces it
        lists.append(eps)
    best, dups = merge(lists)
    mocks = {bool(e.get("mock")) for e in best.values()}
    if len(mocks) > 1:
        sys.exit("refusing to merge mock and real episodes")
    exp = expected_cells(P, a.phase, a.selection)
    missing = [k for k in exp if k not in best]
    extra = [k for k in best if k not in exp]
    pending = [k for k, e in best.items() if e.get("judge_pending")]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for k in exp:
            if k in best:
                f.write(json.dumps(best[k], ensure_ascii=False, default=str) + "\n")
    print(f"merged: {len(best)} unique cells ({len(dups)} duplicate copies dropped); expected {len(exp)}; "
          f"written {len(exp) - len(missing)} to {out}")
    if extra:
        print(f"{len(extra)} cells not in the expected design were NOT written: {extra[:5]}...")
    print(f"crashed-episode reruns are resumed from checkpoints; malformed tool args total: "
          f"{sum(e.get('n_bad_tool_args', 0) for e in best.values())}")
    if pending:
        print(f"{len(pending)} cells finished but NOT judged (run the shard again with --judge-only):")
        for k in pending:
            print("  PENDING", dict(zip(pp.CELL_FIELDS, k)))
    if missing:
        print(f"{len(missing)} MISSING cells:")
        for k in missing:
            print("  MISSING", dict(zip(pp.CELL_FIELDS, k)))
    else:
        print("every expected cell is present")
    sys.exit(1 if (missing or pending) else 0)


if __name__ == "__main__":
    main()
