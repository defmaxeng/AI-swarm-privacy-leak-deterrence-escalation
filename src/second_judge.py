"""Second judge (prereg.md Section 6): run the domain judge with GLM 5.3 Flash on the 200-write sample and report
agreement (Cohen's kappa) with the primary judge on the any-leak verdict, plus per-domain kappa.

  python src/second_judge.py runs/phase2_merged.jsonl --sample results/labels/second_judge_sample.csv \
      --out results/labels/second_judge.csv [--mock]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prereg_params as pp  # noqa: E402
from judge import DOMAINS, judge_domains, write_text  # noqa: E402
from prereg_data import load_episodes  # noqa: E402
from testbed import load_profiles  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    P = pp.load()
    alias = P["prereg"]["second_judge"]["alias"]
    if a.mock:
        import mock
        chat_fn = mock.chat
    else:
        import provider
        chat_fn = provider.chat
    eps = {e["key"]: e for e in load_episodes(a.files)}
    profiles = load_profiles()
    S = pd.read_csv(a.sample)
    done = pd.read_csv(a.out) if Path(a.out).exists() else pd.DataFrame(columns=["key", "widx"])
    done_keys = set(zip(done.key, done.widx))

    def one(r):
        e = eps[r.key]
        w = e["writes"][int(r.widx)]
        j2 = judge_domains(chat_fn, alias, profiles[e["persona"]], write_text(w), f"second_judge:{r.key}",
                           purpose="judge_domain_second")
        return dict(key=r.key, widx=int(r.widx), primary_leak=int(w["judge_domain"]["leak"]),
                    second_leak=int(j2["leak"]), second_parse_ok=j2["parse_ok"],
                    primary_domains=json.dumps(w["judge_domain"]["domains"]), second_domains=json.dumps(j2["domains"]))

    todo = [r for r in S.itertuples() if (r.key, int(r.widx)) not in done_keys]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, todo))
    R = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    R.to_csv(a.out, index=False)
    ok = R[R.second_parse_ok.astype(bool)]
    print(f"{len(R)} writes judged by {alias}; {len(R) - len(ok)} second-judge parse failures (excluded)")
    print(f"any-leak agreement: {(ok.primary_leak == ok.second_leak).mean():.3f}, "
          f"Cohen's kappa {cohen_kappa_score(ok.primary_leak, ok.second_leak):.3f}")
    for d in DOMAINS:
        p1 = ok.primary_domains.map(lambda x: d in json.loads(x))
        p2 = ok.second_domains.map(lambda x: d in json.loads(x))
        if p1.nunique() + p2.nunique() > 2:
            print(f"  {d}: kappa {cohen_kappa_score(p1, p2):.3f} (primary {p1.sum()}, second {p2.sum()})")


if __name__ == "__main__":
    main()
