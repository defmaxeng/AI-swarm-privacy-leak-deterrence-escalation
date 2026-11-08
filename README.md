# Escalation tools vs. swarm bandwagoning

Apart Research Swarm Alignment Hackathon, November 2026.

**Question:** can bandwagoning (peer agents casually oversharing about their humans) get LLM agents past their privacy guardrails, and does a credible "check with my owner before sharing" escalation tool restore them?

This is a reimplementation of the controlled testbed from Priyanshu, Vijay & Pahwa, "Got a Secret? LLM Agents Can't Keep It" (CAIS 2026, arXiv 2605.27766), whose code and data were not released.

## Where things are

| Path | What |
|---|---|
| `plan.md` | The original project brief |
| `prereg.md` | Pre-registration (hypotheses, design, analysis); frozen in git before the main run |
| `STATUS.md` | Running log of every decision, deviation, pilot and failure |
| `ARCHITECTURE.md` | How the testbed works and how it maps to the paper |
| `config/models.yaml` | All models, providers, prices and the judge choice |
| `src/` | Provider layer, testbed, judges, runner, analysis, plots |
| `data/injected_*.json` | The synthetic peer posts and threads (all humans fictional) |
| `runs/*.jsonl` | One line per episode: full transcript, writes, escalations, judge verdicts |
| `logs/calls.jsonl` | Every model call: provider, exact model id, tokens, estimated cost |
| `results/` | Tables and figures generated from `runs/` |

## Reproduce

```bash
bash scripts/get_data.sh                 # third-party data, paper, CIMemories prompts
cp example.env .env                      # add ADAPTION_API_KEY
pip install openai adaption pandas statsmodels matplotlib pyyaml
python src/run.py --exp NAME --model gemma-4-26b-a4b --levels NC ND --mitigations none tool --trials 5
python src/analyze.py runs/NAME.jsonl --out results/NAME
```

Runs are crash-safe: episodes checkpoint after every action and resume when the same command is re-run.

## Provenance rules

- Every number in the report comes from files in `runs/` and `logs/`, produced by the committed code.
- The pre-registration is committed before the main run; the git history shows what was fixed when.
- Third-party data are fetched, not committed. Moltbook snapshot: MIT. CIMemories: CC BY-NC 4.0.
