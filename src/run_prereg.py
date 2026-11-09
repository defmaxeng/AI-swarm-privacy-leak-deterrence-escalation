"""Pre-registered runner (prereg.md Sections 2-3). Sharded and resumable.

  python src/run_prereg.py --phase smoke --out runs/smoke_prereg
  python src/run_prereg.py --phase 1 --out runs/phase1/shard_glm --models glm-5.3-flash
  python src/run_prereg.py --phase 2 --out runs/phase2/shard_a --models glm-5.3-flash qwen3.8-27b
  python src/run_prereg.py --phase 1 --out runs/mock/phase1 --mock        # offline, no API calls

Each shard writes only inside its own --out folder:
  episodes.jsonl  judged episodes (one line per episode; resumable: finished cells are skipped)
  raw.jsonl       every finished episode, written before judging (a judge outage never loses agent work)
  ckpt/           per-episode checkpoints; a crashed episode resumes with the same seed and persona
  errors.log      tracebacks; run_meta.json  start/end times, arguments
Progress lines report only counts (episodes, crashes, malformed tool args, escalation calls), never leakage.
The API key is read from the ADAPTION_API_KEY environment variable (not needed with --mock).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prereg_params as pp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
# arm -> (privacy instruction, escalation tool, flag sentence). none/instr/tool are unchanged from the pilots.
ARMS = {"none": (False, False, False), "instr": (True, False, False), "tool": (False, True, False),
        "flag": (False, False, True), "tool_flag": (False, True, True),
        # Follow-up (NOT pre-registered; deviation D5): instruction + tool
        "instr_tool": (True, True, False)}
_wlock = threading.Lock()


def _append(path: Path, obj: dict):
    with _wlock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if path.exists():
        for line in open(path, encoding="utf-8"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"skipping a damaged line in {path}", flush=True)
    return out


def load_selection(path: Path) -> tuple[str, str, str]:
    sel = json.loads(path.read_text(encoding="utf-8"))
    return sel["control_env"], sel["disclosure_env"], sel["decision"]


def run_one(t, out, chat_fn, platforms, profiles, budget, judge_alias, do_judge, mock):
    from run import judge_episode
    from testbed import run_episode
    redact, tool, flag = ARMS[t["arm"]]
    key = pp.cell_key(t)
    exp = f"prereg_phase{t['phase']}"
    t0 = time.time()
    ckpt = out / "ckpt" / (re.sub(r"[^A-Za-z0-9_.-]", "_", key) + ".json")
    ep = run_episode(chat_fn, t["model"], platforms[(t["environment"], t["seed"])], profiles[t["persona"]],
                     t["persona"], redact, tool, budget, f"{exp}:{key}", checkpoint=ckpt, flag=flag)
    ep.update(t, key=key, exp=exp, mitigation=t["arm"], level=t["environment"], judge=judge_alias, mock=mock,
              wall_s=round(time.time() - t0, 1))
    _append(out / "raw.jsonl", ep)
    try:
        ckpt.unlink()
    except OSError:
        pass
    if do_judge:
        try:
            judge_episode(ep, profiles[t["persona"]], judge_alias, chat_fn=chat_fn)
        except Exception as e:  # judged later with --judge-only
            ep["judge_pending"], ep["judge_error"] = True, repr(e)[:500]
    else:
        ep["judge_pending"], ep["judge_error"] = True, "not judged (--no-judge)"
    return ep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["smoke", "1", "2", "followup"])
    ap.add_argument("--out", required=True, help="this shard's own output folder")
    ap.add_argument("--models", nargs="+", default=None, help="aliases; default = the four prereg models")
    ap.add_argument("--selection", default=str(ROOT / "results/phase1_selection.json"),
                    help="Phase 2 only: output of phase1_select.py")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--part", default=None, help="K/N: run only cells whose index %% N == K (split one model's cells across shards/keys; parts are disjoint and together cover every cell)")
    ap.add_argument("--mock", action="store_true", help="offline mock agent and judge; no API calls")
    ap.add_argument("--no-judge", action="store_true", help="skip judging (default for --phase smoke)")
    ap.add_argument("--judge-only", action="store_true", help="only judge episodes in raw.jsonl not yet judged")
    a = ap.parse_args()

    P = pp.load()
    pr = P["prereg"]
    valid = set(pr["models"].values())
    models = a.models or list(pr["models"].values())
    bad = [m for m in models if m not in valid]
    if bad:
        sys.exit(f"--models {bad} are not prereg models {sorted(valid)}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(out / "episodes.jsonl") + read_jsonl(out / "raw.jsonl")
    if any(bool(e.get("mock")) != a.mock for e in existing):
        sys.exit(f"{out} already holds {'non-' if a.mock else ''}mock episodes; use a separate --out folder")

    if a.mock:
        import mock
        chat_fn = mock.chat
    else:
        if not os.environ.get("ADAPTION_API_KEY"):
            sys.exit("ADAPTION_API_KEY is not set in the environment")
        import provider
        chat_fn = provider.chat
    from testbed import MAX_NUDGES, Platform, load_profiles
    assert MAX_NUDGES == pr["max_nudges"], "testbed.MAX_NUDGES != prereg"
    judge_alias = pr["judge"]["alias"]
    do_judge = not (a.no_judge or a.phase == "smoke")

    if a.phase == "smoke":
        trials = pp.smoke_trials(P, models)
    elif a.phase == "1":
        trials = pp.phase1_trials(P, models)
    elif a.phase == "followup":
        _, disc, decision = load_selection(Path(a.selection))
        print(f"[followup] EXPLORATORY, not pre-registered (D5). environment: {disc} ({decision})")
        trials = pp.followup_trials(P, disc, models)
    else:
        ctrl, disc, decision = load_selection(Path(a.selection))
        print(f"[phase 2] environment pair from {a.selection}: control={ctrl}, disclosure={disc} ({decision})")
        trials = pp.phase2_trials(P, ctrl, disc, models)
    if a.part:
        k, n = map(int, a.part.split("/"))
        assert 0 <= k < n, "--part K/N needs 0 <= K < N"
        trials = [t for i, t in enumerate(trials) if i % n == k]

    done = {e["key"] for e in read_jsonl(out / "episodes.jsonl")}
    raw = {e["key"]: e for e in read_jsonl(out / "raw.jsonl") if e["key"] not in done}
    wanted = {pp.cell_key(t) for t in trials}
    to_judge = [e for k, e in raw.items() if k in wanted] if do_judge else []
    todo = [t for t in trials if pp.cell_key(t) not in done and pp.cell_key(t) not in raw]
    if a.judge_only:
        todo = []
    profiles = load_profiles()
    platforms = {(t["environment"], t["seed"]): None for t in todo}
    for env, seed in platforms:
        platforms[(env, seed)] = Platform(env, seed=seed)

    meta_path = out / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"runs": []}
    run_rec = dict(start=time.strftime("%Y-%m-%d %H:%M:%S"), start_ts=time.time(), argv=sys.argv[1:],
                   n_cells=len(trials), n_todo=len(todo), n_to_judge=len(to_judge), mock=a.mock)
    meta["runs"].append(run_rec)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[phase {a.phase}] {len(trials)} cells in this shard: {len(done)} done, {len(to_judge)} to judge, "
          f"{len(todo)} to run. mock={a.mock}", flush=True)

    counts = dict(ok=0, crashes=0, bad_tool_args=0, escalation_calls=0, eps_with_escalation=0, judge_pending=0)
    t_start = time.time()

    def judge_only(e):
        from run import judge_episode
        try:
            judge_episode(e, profiles[e["persona"]], judge_alias, chat_fn=chat_fn)
            e["judge_pending"], e["judge_error"] = False, None
        except Exception as ex:
            e["judge_pending"], e["judge_error"] = True, repr(ex)[:500]
        return e

    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(judge_only, e): e for e in to_judge}
        futs.update({ex.submit(run_one, t, out, chat_fn, platforms, profiles, pr["budget_tool_calls"], judge_alias,
                               do_judge, a.mock): t for t in todo})
        for f in as_completed(futs):
            t = futs[f]
            try:
                ep = f.result()
            except Exception as e:
                counts["crashes"] += 1
                with _wlock, open(out / "errors.log", "a", encoding="utf-8") as fe:
                    fe.write(f"{pp.cell_key(t)}\n{traceback.format_exc()}\n")
                print(f"CRASH {pp.cell_key(t)}: {e!r}"[:300], flush=True)
                continue
            if ep.get("judge_pending") and do_judge:
                counts["judge_pending"] += 1
                print(f"JUDGE PENDING {ep['key']}: {str(ep.get('judge_error'))[:150]}", flush=True)
                continue  # stays in raw.jsonl; rerun with --judge-only
            counts["ok"] += 1
            counts["bad_tool_args"] += ep.get("n_bad_tool_args", 0)
            counts["escalation_calls"] += len(ep["escalations"])
            counts["eps_with_escalation"] += int(len(ep["escalations"]) > 0)
            _append(out / "episodes.jsonl", ep)
            print(f"  {counts['ok']}/{len(todo) + len(to_judge)} done | crashes {counts['crashes']} | "
                  f"malformed tool args {counts['bad_tool_args']} | escalation calls {counts['escalation_calls']} "
                  f"in {counts['eps_with_escalation']} eps | {(time.time() - t_start) / 60:.1f} min", flush=True)
    run_rec.update(end=time.strftime("%Y-%m-%d %H:%M:%S"), minutes=round((time.time() - t_start) / 60, 1), **counts)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[phase {a.phase}] finished in {run_rec['minutes']} min: {counts}", flush=True)
    if a.phase == "1" and run_rec["minutes"] > pr["phase1"]["time_box_min"]:
        print(f"NOTE: Phase 1 time box of {pr['phase1']['time_box_min']} min overran", flush=True)


if __name__ == "__main__":
    main()
