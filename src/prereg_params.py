"""Load prereg_params.json and assert that it matches prereg.md (the frozen spec wins).

Every value under "prereg" is checked against the text of prereg.md with a pattern tied to the sentence that states
it, and the cue wording in src/testbed.py is checked against prereg.md Appendix A where Appendix A gives it.
Call `load()` at the start of every prereg script; it raises PreregMismatch on any disagreement.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREREG_MD = ROOT / "prereg.md"
PARAMS_JSON = ROOT / "prereg_params.json"


class PreregMismatch(AssertionError):
    pass


def _need(text: str, pattern: str, what: str):
    if not re.search(pattern, text, flags=re.S):
        raise PreregMismatch(f"prereg_params.json disagrees with prereg.md on {what} (pattern {pattern!r} not found)")


def check(P: dict, md: str) -> list[str]:
    """Raise on mismatch. Returns warnings (e.g. Appendix A still has placeholders)."""
    p = P["prereg"]
    e = re.escape
    for name in p["models"]:
        _need(md, r"\*\*Models:\*\*[^\n]*" + e(name), f"model {name}")
    _need(md, r"judge Gemma 4 31B at temperature 0", "judge and temperature")
    _need(md, e(f"domain judge ({p['judge']['name']}, temperature {p['judge']['temperature']})"), "domain judge")
    _need(md, e(f"{p['second_judge']['name']} on {p['second_judge']['n_writes']} random writes"), "second judge")
    _need(md, e(f"{p['budget_tool_calls']} tool-call budget, up to {p['max_nudges']} nudges"), "budget and nudges")
    _need(md, e(f"one of the {p['n_domains']} privacy domains"), "number of domains")
    a = p["phase1"]
    _need(md, e(f"Seeds {a['seed_min']} to {a['seed_max']}"), "Phase 1 seeds")
    _need(md, e(f"Time box: {a['time_box_min']} minutes"), "time box")
    _need(md, e(f"4 models × {a['control_episodes_per_model']} episodes = {a['control_total']}"), "control episodes")
    _need(md, e(f"4 models × {a['variant_episodes_per_model']} episodes = {a['variant_total']}. Total {a['total']}"),
          "variant episodes")
    _need(md, e(f"Gap is at least **{a['gap_threshold_pp']}** percentage points"), "gap threshold")
    _need(md, e(f"Control RLR is below **{a['control_rlr_max_pct']}%**"), "control threshold")
    _need(md, e(f"control-environment RLR in Phase 1 is at least {a['ceiling_control_rlr_pct']}%"), "ceiling")
    _need(md, e(f"gaps are within {a['tie_within_pp']} points, use the simpler one (A before B before C)"), "tie rule")
    _need(md, e(f'label the testbed "{a["fallback"]["label"]}') + r'.?"', "fallback label")
    _need(md, r"\| A \| 5 of 5 disclose, same level of detail as the pilots \|", "variant A")
    _need(md, r"\| B \| 4 of 5 disclose, each naming a specific value", "variant B")
    _need(md, r"\| C \| 5 of 5 disclose, each naming a specific value", "variant C")
    _need(md, r"no-mitigation arm only", "Phase 1 arm = none")
    b = p["phase2"]
    _need(md, r"\*\*Arms \(5\):\*\*", "five arms")
    for arm in b["arms"]:
        _need(md, r"\n\| " + e(arm) + r" \|", f"arm {arm}")
    _need(md, e(f"5 arms × 2 environments × 4 models × {b['episodes_per_cell']} episodes = {b['total']} episodes. "
                f"Seeds {b['seed_min']} and up"), "Phase 2 size and seeds")
    s = p["smoke"]
    _need(md, e(f"One episode per model in the tool and tool_flag arms ({s['total']} episodes)"), "smoke test")
    an = p["analysis"]
    _need(md, e(f"Holm correction at {an['alpha']}"), "alpha and Holm")
    _need(md, e(f"{an['ci_pct']}% episode-clustered bootstrap intervals ({an['bootstrap_resamples']:,} resamples)"),
          "bootstrap")
    _need(md, e(f"text similarity of at least {an['bypass_similarity']} to the held draft (sensitivity checks at "
                f"{an['bypass_similarity_sensitivity'][0]} and {an['bypass_similarity_sensitivity'][1]})"), "bypass")
    for h in an["confirmatory"]:
        _need(md, e(f"**{h['id']}:** {h['treat']} has a lower PLR than {h['ref']}"), h["id"])
    lab = p["labels"]
    _need(md, e(f"{lab['n_total']} writes ({lab['n_flagged']} flagged and {lab['n_unflagged']} unflagged"), "labels")

    # Cue wording: code vs Appendix A.
    import testbed
    warnings = []
    appx = md.split("## Appendix A", 1)[1] if "## Appendix A" in md else ""
    for label, code_text in [("Tool description", testbed.ASK_OWNER_DESC), ("Flag sentence", testbed.FLAG_SENTENCE),
                             ("Instruction paragraph", testbed.REDACT_INSTRUCTION.strip())]:
        m = re.search(r"\*\*" + label + r":\*\*\s*(.+)", appx)
        val = m.group(1).strip() if m else ""
        if not val or val.startswith("[paste") or "fill" in val.lower():
            warnings.append(f"Appendix A '{label}' is a placeholder in prereg.md; code uses: {code_text!r}")
        elif val.startswith("[draft:"):
            q = re.search(r'"(.+)"', val)
            if not q or q.group(1) != code_text:
                raise PreregMismatch(f"Appendix A draft '{label}' differs from code: {val!r} vs {code_text!r}")
            warnings.append(f"Appendix A '{label}' is still marked [draft] in prereg.md")
        else:
            q = re.search(r'"(.+)"', val)
            norm = lambda x: re.sub(r"\s+", " ", x.replace("## Privacy", "")).strip()  # noqa: E731
            if norm(q.group(1) if q else val) != norm(code_text):
                raise PreregMismatch(f"Appendix A '{label}' differs from code: {val!r} vs {code_text!r}")
    return warnings


def load(verbose: bool = True) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    P = json.loads(PARAMS_JSON.read_text(encoding="utf-8"))
    warnings = check(P, PREREG_MD.read_text(encoding="utf-8"))
    if verbose:
        for w in warnings:
            print("PREREG WARNING:", w, flush=True)
    return P


# ---------------------------------------------------------------------------
# Seeds and the persona schedule (prereg.md Sec. 2-3: seeds 1000-1999 for Phase 1, 2000+ for Phase 2).
def persona_for_seed(seed: int, P: dict) -> int:
    """One seeded schedule shared by every arm, environment and model: the same seed always gets the same persona.
    Within each block of 10 consecutive seeds every persona appears once."""
    block, i = divmod(seed, 10)
    perm = list(range(10))
    random.Random(P["implementation"]["persona_schedule_seed"] * 100003 + block).shuffle(perm)
    return perm[i]


def phase1_trials(P: dict, models=None) -> list[dict]:
    a = P["prereg"]["phase1"]
    models = models or list(P["prereg"]["models"].values())
    cells = [(a["control_env"], "control", a["control_episodes_per_model"])]
    cells += [(a["variants"][v], f"variant_{v}", a["variant_episodes_per_model"]) for v in a["variant_order"]]
    out = []
    for i in range(max(n for _, _, n in cells)):
        seed = a["seed_min"] + i
        assert seed <= a["seed_max"]
        for m in models:
            for env, role, n in cells:
                if i < n:
                    out.append(dict(phase="1", model=m, environment=env, env_role=role, arm=a["arm"], seed=seed,
                                    persona=persona_for_seed(seed, P)))
    return out


def phase2_trials(P: dict, control_env: str, disclosure_env: str, models=None) -> list[dict]:
    b = P["prereg"]["phase2"]
    models = models or list(P["prereg"]["models"].values())
    out = []
    for i in range(b["episodes_per_cell"]):
        seed = b["seed_min"] + i
        for m in models:
            for env, role in [(control_env, "control"), (disclosure_env, "disclosure")]:
                for arm in b["arms"]:
                    out.append(dict(phase="2", model=m, environment=env, env_role=role, arm=arm, seed=seed,
                                    persona=persona_for_seed(seed, P)))
    return out


def smoke_trials(P: dict, models=None) -> list[dict]:
    s, im = P["prereg"]["smoke"], P["implementation"]
    models = models or list(P["prereg"]["models"].values())
    return [dict(phase="smoke", model=m, environment=im["smoke_env"], env_role="smoke", arm=arm,
                 seed=im["smoke_seed"], persona=persona_for_seed(im["smoke_seed"], P))
            for arm in s["arms"] for m in models]


CELL_FIELDS = ("model", "persona", "environment", "arm", "seed")


def cell_key(t: dict) -> str:
    return f"{t['model']}|{t['environment']}|{t['arm']}|p{t['persona']}|s{t['seed']}"


if __name__ == "__main__":
    P = load()
    print("prereg_params.json matches prereg.md")
    print("Phase 1:", len(phase1_trials(P)), "episodes; Phase 2:", len(phase2_trials(P, "NC", "ND")),
          "episodes; smoke:", len(smoke_trials(P)))
