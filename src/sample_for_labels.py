"""Judge-validation samples (prereg.md Section 6). Reads no outcome beyond the domain-judge flag needed to stratify.

  python src/sample_for_labels.py runs/phase2_merged.jsonl --out results/labels

Writes:
  labels_blank.csv          50 posted writes (25 domain-judge-flagged, 25 unflagged), shuffled. Columns: item_id,
                            owner_profile, message, and blank label columns. NO arm, model, environment or judge
                            columns. Give one copy to each labeller.
  labels_key.csv            item_id -> episode key, write index, model, arm, environment, domain-judge verdict.
                            Keep away from the labellers.
  second_judge_sample.csv   200 random posted writes (episode key + write index) for the second judge
                            (src/second_judge.py).
Stratification: within the flagged and unflagged pools, writes are drawn round-robin over model x arm strata (in a
seeded random order), so each stratum contributes as evenly as its pool allows.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prereg_params as pp  # noqa: E402
from judge import profile_by_domain  # noqa: E402
from prereg_data import load_episodes, writes_df  # noqa: E402
from testbed import load_profiles  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def stratified(pool: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    strata = {}
    for i, r in pool.iterrows():
        strata.setdefault((r.model, r.arm), []).append(i)
    for v in strata.values():
        rng.shuffle(v)
    keys = sorted(strata)
    rng.shuffle(keys)
    picked = []
    while len(picked) < n and any(strata[k] for k in keys):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop())
    if len(picked) < n:
        print(f"WARNING: only {len(picked)} writes available, {n} requested")
    return pool.loc[picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=str(ROOT / "results/labels"))
    a = ap.parse_args()
    P = pp.load()
    lab, im = P["prereg"]["labels"], P["implementation"]
    eps = load_episodes(a.files)
    mock = any(e.get("mock") for e in eps)
    out = Path(a.out)
    if mock and out.resolve().is_relative_to((ROOT / "results").resolve()):
        sys.exit("refusing to write mock samples into results/; pass --out elsewhere")
    out.mkdir(parents=True, exist_ok=True)
    W = writes_df(eps)
    profiles = load_profiles()
    rng = random.Random(im["label_sample_seed"])
    flagged = stratified(W[W.leak == 1], lab["n_flagged"], rng)
    unflagged = stratified(W[W.leak == 0], lab["n_unflagged"], rng)
    S = pd.concat([flagged, unflagged]).sample(frac=1, random_state=im["label_sample_seed"]).reset_index(drop=True)
    S["item_id"] = [f"L{i + 1:03d}" for i in range(len(S))]
    blank = pd.DataFrame(dict(
        item_id=S.item_id,
        owner_profile=[profile_by_domain(profiles[p]) for p in S.persona],
        message=S.text,
        leak_yes_no="", domains_leaked="", notes=""))
    blank.to_csv(out / "labels_blank.csv", index=False, encoding="utf-8")
    S[["item_id", "key", "widx", "model", "arm", "environment", "env_role", "leak", "domains"]].rename(
        columns={"leak": "domain_judge_leak", "domains": "domain_judge_domains"}).to_csv(
        out / "labels_key.csv", index=False, encoding="utf-8")
    n2 = P["prereg"]["second_judge"]["n_writes"]
    S2 = W.sample(n=min(n2, len(W)), random_state=im["second_judge_sample_seed"])
    S2[["key", "widx", "model", "arm", "environment"]].to_csv(out / "second_judge_sample.csv", index=False)
    print(f"labels: {len(S)} items ({int(S.leak.sum())} flagged, {int((S.leak == 0).sum())} unflagged); "
          f"strata model x arm: {S.groupby(['model', 'arm']).size().to_dict()}")
    print(f"second-judge sample: {len(S2)} writes")
    print(f"wrote {out}/labels_blank.csv, labels_key.csv, second_judge_sample.csv")


if __name__ == "__main__":
    main()
