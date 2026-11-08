"""Experiment runner. Resumable: trials already in the output file are skipped.

Example:
  python src/run.py --exp phase1 --model gpt-5-nano --levels 1 5 --mitigations none --trials 30 --budget 20
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
import provider  # noqa: E402
from judge import judge_domains, judge_text, write_text  # noqa: E402
from testbed import Platform, load_profiles, run_episode  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MITIGATIONS = {"none": (False, False), "instr": (True, False), "tool": (False, True), "instr_tool": (True, True)}

_wlock = threading.Lock()


def trial_key(t):
    return f"{t['model']}|{t['level']}|{t['mitigation']}|p{t['persona']}|r{t['rep']}"


def make_trials(models, levels, mitigations, trials):
    out = []
    for rep_block in range((trials + 9) // 10):
        for p in range(10):
            if rep_block * 10 + p >= trials:
                break
            for m in models:
                for lv in levels:
                    for mit in mitigations:
                        out.append(dict(model=m, level=lv, mitigation=mit, persona=p, rep=rep_block))
    return out  # interleaved across models and cells so partial runs stay balanced


def run_trial(t, platforms, profiles, budget, judge_alias, exp):
    redact, tool = MITIGATIONS[t["mitigation"]]
    rid = f"{exp}:{trial_key(t)}"
    t0 = time.time()
    # Checkpoint after every model turn and tool call; a re-run resumes mid-episode from here.
    ckpt = ROOT / "runs" / f"{exp}_ckpt" / (re.sub(r"[^A-Za-z0-9_.-]", "_", trial_key(t)) + ".json")
    ep = run_episode(provider.chat, t["model"], platforms[t["level"]], profiles[t["persona"]], t["persona"],
                     redact, tool, budget, rid, checkpoint=ckpt)
    ep.update(t, key=trial_key(t), exp=exp, judge=judge_alias, wall_s=round(time.time() - t0, 1))
    # Save the agent episode BEFORE judging, so a judge outage never loses agent work.
    with _wlock, open(ROOT / "runs" / f"{exp}_raw.jsonl", "a", encoding="utf-8") as fr:
        fr.write(json.dumps(ep, ensure_ascii=False, default=str) + "\n")
        fr.flush()
        os.fsync(fr.fileno())
    try:
        ckpt.unlink()  # the finished episode is now safely in the raw file
    except OSError:
        pass
    try:
        judge_episode(ep, profiles[t["persona"]], judge_alias)
    except Exception as e:  # judged later by src/rejudge.py
        ep["judge_pending"] = True
        ep["judge_error"] = repr(e)[:500]
    return ep


def judge_episode(ep, prof, judge_alias, chat_fn=None):
    chat_fn = chat_fn or provider.chat
    rid = f"{ep['exp']}:{ep['key']}"
    for w in ep["writes"]:
        if not w.get("judge"):
            w["judge"] = judge_text(chat_fn, judge_alias, prof, write_text(w), rid)
        if not w.get("judge_domain"):
            w["judge_domain"] = judge_domains(chat_fn, judge_alias, prof, write_text(w), rid)
    for e in ep["escalations"]:
        if not e.get("judge"):
            e["judge"] = judge_text(chat_fn, judge_alias, prof, e["draft"], rid, purpose="judge_draft")
        if not e.get("judge_domain"):
            e["judge_domain"] = judge_domains(chat_fn, judge_alias, prof, e["draft"], rid,
                                              purpose="judge_domain_draft")
    ep.update(judge_pending=False, judge_error=None,
              leak=any(w["judge"]["leak"] for w in ep["writes"]),
              n_leaky_writes=sum(w["judge"]["leak"] for w in ep["writes"]),
              leak_domain=any(w["judge_domain"]["leak"] for w in ep["writes"]),
              n_leaky_writes_domain=sum(w["judge_domain"]["leak"] for w in ep["writes"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--levels", nargs="+", default=["TC", "TD"], help="environments: L1..L5, TC, TD")
    ap.add_argument("--mitigations", nargs="+", default=list(MITIGATIONS))
    ap.add_argument("--trials", type=int, default=10, help="trials per cell (multiple of 10 = full persona sweeps)")
    ap.add_argument("--budget", type=int, default=40, help="tool-call budget B")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--judge", default=provider.CFG["judge"]["primary"])
    ap.add_argument("--max-spend", type=float, default=None, help="stop this run once global spend passes this")
    a = ap.parse_args()

    out = ROOT / "runs" / f"{a.exp}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["key"])
            except Exception:
                pass
    profiles = load_profiles()
    platforms = {lv: Platform(lv) for lv in a.levels}
    trials = [t for t in make_trials(a.model, a.levels, a.mitigations, a.trials) if trial_key(t) not in done]
    # Episodes saved raw (agent finished) but never written judged: judge them instead of re-running the agent.
    raw_path = ROOT / "runs" / f"{a.exp}_raw.jsonl"
    raw_eps = {}
    if raw_path.exists():
        for line in open(raw_path, encoding="utf-8"):
            try:
                e = json.loads(line)
            except Exception:
                continue
            if e["key"] not in done:
                raw_eps[e["key"]] = e
    wanted = {trial_key(t) for t in trials}
    raw_eps = {k: e for k, e in raw_eps.items() if k in wanted}
    trials = [t for t in trials if trial_key(t) not in raw_eps]
    print(f"[{a.exp}] {len(raw_eps)} saved episodes to judge without re-running", flush=True)
    print(f"[{a.exp}] {len(trials)} trials to run ({len(done)} already done). spend so far ${provider.spent():.4f}",
          flush=True)
    cap = a.max_spend or provider.CFG["budget"]["hard_stop_usd"]
    stop = threading.Event()
    n_ok = n_err = 0

    def job(t):
        if stop.is_set() or provider.spent() >= cap:
            stop.set()
            return None
        return run_trial(t, platforms, profiles, a.budget, a.judge, a.exp)

    def judge_only(e):
        e["judge"] = a.judge
        try:
            judge_episode(e, profiles[e["persona"]], a.judge)
        except Exception as ex:
            e["judge_pending"] = True
            e["judge_error"] = repr(ex)[:500]
        return e

    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(judge_only, e): dict(model=e["model"], level=e["level"], mitigation=e["mitigation"],
                                              persona=e["persona"], rep=e["rep"]) for e in raw_eps.values()}
        futs.update({ex.submit(job, t): t for t in trials})
        for f in as_completed(futs):
            t = futs[f]
            try:
                ep = f.result()
            except provider.BudgetExceeded as e:
                stop.set()
                print("BUDGET STOP:", e, flush=True)
                continue
            except Exception as e:
                n_err += 1
                with _wlock, open(ROOT / "logs" / f"{a.exp}_errors.log", "a", encoding="utf-8") as fe:
                    fe.write(f"{trial_key(t)}\n{traceback.format_exc()}\n")
                print(f"ERR {trial_key(t)}: {e!r}"[:300], flush=True)
                continue
            if ep is None:
                continue
            n_ok += 1
            with _wlock, open(out, "a", encoding="utf-8") as fo:
                fo.write(json.dumps(ep, ensure_ascii=False, default=str) + "\n")
                fo.flush()
                os.fsync(fo.fileno())
            if ep.get("judge_pending"):
                print(f"JUDGE PENDING {trial_key(t)}: {ep['judge_error'][:150]}", flush=True)
            if n_ok % 10 == 0:
                print(f"  {n_ok} done, {n_err} errors, spend ${provider.spent():.4f}", flush=True)
    print(f"[{a.exp}] finished: {n_ok} ok, {n_err} errors, stopped={stop.is_set()}, spend ${provider.spent():.4f}",
          flush=True)


if __name__ == "__main__":
    main()
