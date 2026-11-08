"""Pilot figure: share of an agent's posts/replies that leak, per model, control vs peer-disclosure threads.
Usage: python src/plot_pilot.py runs/pilot4_threads.jsonl --out results/pilot4
Writes are not independent (clustered within episodes), so the Wilson intervals here are optimistic."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

ENV = {"TC": ("Control threads (no peer disclosure)", "#2a78d6"),
       "TD": ("Peer-disclosure threads", "#eb6834"),
       "NC": ("Control threads (no peer disclosure)", "#2a78d6"),
       "ND": ("Peer-disclosure threads", "#eb6834")}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def load(path):
    eps = {}
    for line in open(path, encoding="utf-8"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not e.get("judge_pending"):
            eps[e["key"]] = e
    return list(eps.values())


def rate(eps, field):
    W = [w for e in eps for w in e["writes"]]
    k = sum(bool(w[field]["leak"]) for w in W)
    lo, hi = proportion_confint(k, len(W), method="wilson") if W else (0, 0)
    return k, len(W), (k / len(W) if W else 0), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--out", default="results/pilot4")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = load(a.file)
    models = sorted({e["model"] for e in eps})
    rows = models + ["All models"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(rows) + 2.2), sharey=True)
    lines = ["model,judge,env,episodes,writes,leaky_writes,rate,ci_lo,ci_hi"]
    for ax, (field, jname) in zip(axes, [("judge_domain", "Domain judge (any owner detail)"),
                                         ("judge", "Strict judge (exact attribute value)")]):
        for i, m in enumerate(rows):
            y = len(rows) - 1 - i
            for j, env in enumerate(["TC", "TD"]):
                sub = [e for e in eps if e["level"] == env and (m == "All models" or e["model"] == m)]
                if not sub:
                    continue
                k, n, r, lo, hi = rate(sub, field)
                lines.append(f"{m},{field},{env},{len(sub)},{n},{k},{r:.3f},{lo:.3f},{hi:.3f}")
                yy = y + (0.16 if env == "TC" else -0.16)
                ax.plot([lo * 100, hi * 100], [yy, yy], color=ENV[env][1], lw=2, solid_capstyle="round")
                ax.plot(r * 100, yy, "o", ms=8, color=ENV[env][1], mec="white", mew=1.5,
                        label=ENV[env][0] if i == 0 else None)
        ax.axhline(0.5, color=GRID, lw=1)
        ax.set_xlim(0, 100)
        ax.set_title(jname, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Posts and replies that leak owner information (%)", fontsize=9.5, color=MUTED)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(list(reversed(rows)), fontsize=10, color=INK)
    axes[0].get_yticklabels()[0].set_fontweight("bold")
    n_eps = len(eps)
    fig.suptitle(f"Agents leak in most posts, with or without peers oversharing (pilot, {n_eps} episodes, no mitigation)",
                 fontsize=12.5, color=INK, x=0.01, ha="left", fontweight="bold")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.01, 0.93), ncol=2,
               frameon=False, fontsize=9.5)
    fig.text(0.01, 0.005, "Dots: share of writes judged leaking. Lines: Wilson 95% CI over writes (optimistic: writes cluster "
             "within episodes). 2 episodes per model per condition. Judge: Gemma 4 31B.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    fig.savefig(out / "pilot_leak_rates.png", dpi=180, facecolor="white")
    (out / "pilot_leak_rates.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out / "pilot_leak_rates.png", "and .csv")


if __name__ == "__main__":
    main()
