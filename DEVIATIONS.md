# Deviations from prereg.md

Every departure from `prereg.md` after it is frozen is logged here: time (ET), what changed, why, and whether any outcome data (Phase 1 or Phase 2 leakage) had been seen at the time. `prereg.md` itself is never edited.

| ID | Time (ET) | What changed | Why | Outcome data seen? |
|---|---|---|---|---|
| D2 | 2026-11-08 18:11-19:59 | Phase 1 ran 108 min, overrunning the 60-min time box (Sec. 2). All 60 pre-registered episodes were completed rather than stopping at 60 min. | Adaption per-user rate limit (HTTP 429) made episodes ~4-10x slower than model latency alone. Prereg does not say what to do on overrun; stopping would leave cells empty. | No. Overrun was clear by ~18:31; no leakage was computed until all 60 episodes were judged (19:59). |
| D3 | 2026-11-08 18:33 and 19:23 | Concurrency changed mid-run: 4 shards x 4 workers -> 4 x 2 (18:33) -> GLM and DeepSeek shards 3 workers each (19:23). Processes were stopped and restarted; in-progress episodes resumed from checkpoints. 7 episodes that failed after 10 rate-limit retries (429) were rerun from their checkpoints with the same seed and persona (Sec. 3 crash rule). | Rate limiting; throughput. Operational only; no change to design, seeds, models, judge or prompts. | No. |

Planned deviation D1 (prereg.md Sec. 3): if the smoke test shows 0 tool calls in 8 episodes, the tool description may be revised once.
