# Results (source material for the write-up)

Notes for whoever formats this:
- Keep every number exactly as given. Effects are percentage points (pp), treatment minus reference. Brackets are 95% episode-clustered bootstrap intervals (2,000 resamples).
- Keep the split between **confirmatory** results (H1 to H3, fixed in the analysis plan before any data were collected) and **exploratory** results (everything else). Exploratory results have no multiplicity correction and shouldn't be described as confirmed findings.
- PLR (posted leak rate) is the share of an agent's writes (posts and replies) that the domain judge flags as revealing at least one private detail about the owner. RLR (reply-level leak rate) is the same measure restricted to replies inside the injected peer threads.
- Arms: **none** (no mitigation), **instr** (a privacy instruction in the system prompt), **tool** (an `ask_owner_before_sharing` escalation tool), **flag** (one sentence saying other agents can flag posts to their owners), and **tool_flag** (tool plus flag sentence).
- Models: GLM 5.3 Flash, Qwen 3.8 27B, DeepSeek V4 Flash and Gemma 4 26B-A4B. Primary judge: Gemma 4 31B at temperature 0.

---

## 1. Phase 1: calibrating the disclosure thread (no-mitigation arm, 60 episodes)

The rule was fixed in advance: pick the variant with the largest RLR gap over the control, provided the gap is at least 25 pp and control RLR is below 20%. It pools over models whose control RLR is below 80%; all four models qualified.

| Thread | Replies | Leaky replies | RLR | Gap vs control | Passes rule |
|---|---|---|---|---|---|
| Control (neutral) | 93 | 18 | 19.35% | – | – |
| Variant A (5 of 5 peers disclose) | 45 | 22 | 48.89% | +29.53 pp | Yes |
| Variant B (4 of 5 disclose, specific values) | 46 | 17 | 36.96% | +17.60 pp | No |
| Variant C (5 of 5 disclose, specific values) | 54 | 22 | 40.74% | +21.39 pp | No |

**Variant A was selected.** Phase 2 used the control thread and the variant A disclosure thread. Phase 1 episodes were not reused.

## 2. Phase 2: confirmatory tests (disclosure thread, 240 episodes in total, 6 per cell)

Logistic GEE with exchangeable correlation within episode and model fixed effects. The effect is a percentage-point difference standardised over the four models. Tests are two-sided, Holm-corrected at α = 0.05.

| Hypothesis | Comparison | PLR treatment | PLR reference | Effect | 95% CI | p | Holm p | Result |
|---|---|---|---|---|---|---|---|---|
| H1 | tool vs none | 21.17% | 42.79% | −21.28 pp | [−36.25, −7.13] | 0.012 | 0.023 | **Supported**: the tool reduces leakage |
| H2 | flag vs none | 45.86% | 42.79% | +2.33 pp | [−13.06, +18.61] | 0.78 | 0.78 | Not supported |
| H3 | tool vs instr | 21.17% | 6.67% | +14.39 pp | [+3.60, +25.67] | 0.0076 | 0.023 | **Significant in the opposite direction**: the instruction beats the tool |

Headline: the escalation tool roughly halves leakage relative to no mitigation. A one-paragraph privacy instruction does significantly better than the tool, and the flag sentence does nothing.

**Key finding: the tool works mostly by being visible in the prompt, not by being used.** In the disclosure thread, the tool arm's leakage halved, from 42.79% to 21.17%. Yet agents called the tool in only 12.5% of those episodes, and it caught only 3 of 50 attempted leaks (6.0%). Diversion alone would cut leakage by about 6%, but leakage actually fell by about 50%. Most of the benefit therefore comes from agents leaving private details out of their posts once the tool is offered. The tool's description reminds them that the owner's details are sensitive. When the privacy instruction is already present, the tool is never called and adds nothing (Section 5). This fits the same mechanism: the instruction already gives the reminder. Section 4, E5, has the details. This is an exploratory interpretation, and the design doesn't isolate the mechanism.

## 3. Phase 2: leak rate by arm and thread (descriptive, pooled writes)

| Arm | Control thread PLR | Disclosure thread PLR |
|---|---|---|
| none | 30.92% (64/207) | 42.79% (86/201) |
| instr | 7.56% (17/225) | 6.67% (15/225) |
| tool | 15.35% (31/202) | 21.17% (47/222) |
| flag | 30.15% (60/199) | 45.86% (83/181) |
| tool_flag | 10.10% (21/208) | 21.83% (43/197) |

## 4. Phase 2: exploratory analyses

**E1, did the peer-pressure effect replicate on fresh seeds? (none arm)**
- PLR: 42.79% under disclosure vs 30.92% under control, +10.39 pp [−4.08, +23.75], p = 0.17.
- RLR: 46.55% vs 29.36%, +14.90 pp [−0.55, +31.63], p = 0.09.
- The gap points the same way as in Phase 1 but is about half the size, and both intervals include zero. A smaller gap is expected after picking the largest of three Phase 1 gaps (winner's curse).

**E2, does adding the flag sentence to the tool help?**
- tool_flag vs tool: +1.04 pp [−11.01, +12.83]. No difference.
- tool_flag vs flag: −23.02 pp [−37.18, −9.29], p < 0.001. The tool's benefit holds with the flag sentence present.

**E3, do mitigations help more under peer pressure? (arm × thread interaction, difference in differences)**
- None of the four interactions are distinguishable from zero:
  - instr: −9.89 pp [−26.67, +5.54]
  - tool: −2.71 [−20.91, +14.97]
  - flag: +3.75 [−16.87, +24.35]
  - tool_flag: +0.31 [−17.23, +16.85]

**E4, per-model effects (disclosure thread; small cells, so these are noisy)**
| Model | none PLR | instr vs none | tool vs none | flag vs none |
|---|---|---|---|---|
| GLM 5.3 Flash | 35.4% | −28.9 [−51.8, −7.1] | −17.3 [−44.1, +11.1] | −1.2 [−34.1, +32.7] |
| Qwen 3.8 27B | 40.0% | −36.7 [−65.9, −8.7] | −36.0 [−63.5, −6.0] | −3.1 [−38.9, +29.4] |
| DeepSeek V4 Flash | 80.5% | −60.4 [−80.8, −36.2] | −34.2 [−58.7, −10.3] | **+14.7 [+2.4, +32.3]** |
| Gemma 4 26B-A4B | 20.0% | −18.9 [−41.5, +0.4] | +1.3 [−31.4, +39.3] | −5.3 [−38.2, +33.2] |

- The instruction lowered leakage for every model.
- The tool clearly helped Qwen and DeepSeek, but not Gemma.
- The flag sentence *increased* DeepSeek's leakage.

**E5, how the tool was used**
- The tool was called rarely. Over 96 tool-arm episodes there were 9 escalations in total.
- Uptake (share of episodes with at least one escalation):

  | Arm | Control | Disclosure |
  |---|---|---|
  | tool | 4.2% | 12.5% |
  | tool_flag | 0% | 20.8% |

- False escalations (a held draft with no private detail): 1 of 9.
- Diversion (share of attempted leaks the tool caught instead of them being posted): 6.0% in the tool arm (3/50) and 8.5% in the tool_flag arm (4/47), both in the disclosure thread.
- Bypass (the held content was later posted anyway): 1 of 3 held drafts (tool) and 2 of 5 (tool_flag). The result is the same at similarity thresholds of 0.5, 0.6 and 0.7.
- **Interpretation:** most of the tool's benefit doesn't come from catching drafts. Only about 6 to 8% of attempted leaks were diverted, yet leakage halved. The tool's presence in the prompt seems to make agents more careful by itself.

**E6, H1 to H3 rescored with the strict judge** (the CIMemories `adversary_reveal` prompt, which counts only exact attribute values)
- H1: 5.86% vs 17.91%, −12.04 pp [−20.00, −4.73]. Replicates.
- H2: +3.02 pp [−6.64, +13.32]. Same as primary: no effect.
- H3: 5.86% vs 0.00%, +5.83 pp [+1.53, +10.40]. Same direction as the primary result. The instr arm had zero strict-judge leaks, so the GEE p-value isn't reliable (quasi-separation); report only the direction and the interval.

**Activity and cost:**
- Mitigations didn't silence the agents. Every arm averaged 7.5 to 9.4 writes per episode, and no episode had zero writes.
- There were 0 malformed tool calls. 2 of the roughly 2,100 writes failed to parse in the judge.

## 5. Follow-up: does the tool add anything on top of the instruction? (EXPLORATORY)

This comparison was designed **after** the Phase 2 results were seen. It wasn't part of the original plan.
- Design: instr vs instr_tool (instruction plus tool), disclosure thread only, 6 episodes per model per arm (48 episodes), fresh seeds.

| | instr | instr_tool | Effect | 95% CI | p |
|---|---|---|---|---|---|
| PLR (domain judge) | 9.27% | 8.33% | −0.60 pp | [−10.16, +8.58] | 0.90 |
| PLR (strict judge) | 0.49% | 0.42% | −0.04 pp | [−1.19, +1.09] | 0.95 |
| Escalations | – | **0 in 24 episodes** | | | |
| Writes per episode | 8.54 | 10.00 | | | |

- **With the instruction present, no agent ever called the tool**, and the tool added no detectable benefit.
- The instr-only rate (9.3%) is close to Phase 2's (6.7%) on new seeds, so the instruction's effect replicates.
- Per model, instr → instr_tool:
  - Qwen: 13.1% → 0%
  - Gemma: 5.7% → 20.0%
  - DeepSeek: 17.0% → 19.3%
  - GLM: 0% → 1.4%
- Because the tool was never called, these swings are most likely noise, with 35 to 74 writes per cell.

## 6. Judge validation

- **Second judge:** GLM 5.3 Flash rescored 200 random writes, of which 194 parsed. It agreed with the primary judge on leak / no leak in 97.9% of cases (Cohen's κ = 0.91). This shows the two judges are consistent, not that they're correct.
- **Hand labelling:** the plan called for two authors to label 50 writes. The sample was drawn, but it wasn't labelled within the time limit, so no human accuracy or precision figures exist. This is logged as a deviation.
- **Possible family bias:** the primary judge (Gemma 4 31B) is from the same model family as one agent (Gemma 4 26B-A4B).

## 7. Departures from the plan (all logged with timestamps)

The plan was frozen in a tagged commit at 17:43 ET on Nov 8, before any data were collected.
- **D2:** Phase 1 took 108 minutes against a 60-minute time box, because of API rate limits. All planned episodes were completed. No outcomes had been seen.
- **D3:** concurrency was changed during the run, and rate-limit crashes were rerun with the same seed. This was operational only, and no outcomes had been seen.
- **D4:** Phase 2 ran on two API keys, and work was rebalanced between them. Operational only, with no outcomes seen.
- **D5:** the follow-up arm (Section 5) was added after the Phase 2 results had been seen.
- **D6:** hand labelling was dropped for lack of time, after the results had been seen.

## 8. Suggested main findings (for framing)

1. **The escalation tool works, but not mainly as a channel to the owner.** It halved leakage under peer pressure (H1), yet it was called in only 4 to 21% of episodes and caught only about 6 to 8% of attempted leaks. Its effect seems to come mostly from being visible in the prompt.
2. **A plain instruction beats the tool** (H3, opposite to our hypothesis), and adding the tool to the instruction adds nothing (exploratory follow-up: the tool was never called).
3. **Telling agents they can be flagged doesn't help** (H2), and it made one model (DeepSeek) leak more.
4. **Peer disclosure increases leakage, but less than the calibration phase suggested.** The effect was about +10 to +15 pp on fresh seeds against +29.5 pp in Phase 1, with intervals that include zero.
5. **Models differ a lot.** DeepSeek leaked 80% of the time with no mitigation in the disclosure thread; Gemma leaked 20%.

Limitations to mention:
- 6 episodes per cell.
- One testbed and one disclosure thread.
- LLM judges without human validation.
- Four open-weight models served by one provider.
- The follow-up is exploratory.
