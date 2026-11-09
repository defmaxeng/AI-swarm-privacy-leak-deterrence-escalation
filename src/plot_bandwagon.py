"""Bandwagoning pilot figure: write-level leak rate, control vs peer-disclosure threads,
inviting topics (pilot 4: TC/TD) next to neutral topics (pilot 5: NC/ND). No mitigation.
Usage: python src/plot_bandwagon.py runs/pilot4_threads.jsonl runs/pilot5_neutral.jsonl --out results/bandwagon_pilot
Writes are clustered within episodes, so the Wilson intervals here are optimistic."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_pilot import load, rate, INK, MUTED, GRID

CTRL, DISC = "#2a78d6", "#eb6834"


def panel(ax, eps, envs, title, field):
    models = sorted({e["model"] for e in eps})
    rows = models + ["All models"]
    out = []
    for i, m in enumerate(rows):
        y = len(rows) - 1 - i
        for env, col, off, lab in [(envs[0], CTRL, 0.16, "Control threads (no peer disclosure)"),
                                   (envs[1], DISC, -0.16, "Peer-disclosure threads")]:
            sub = [e for e in eps if e["level"] == env and (m == "All models" or e["model"] == m)]
            if not sub:
                continue
            k, n, r, lo, hi = rate(sub, field)
            out.append((m, field, env, len(sub), n, k, r, lo, hi))
            ax.plot([lo * 100, hi * 100], [y + off] * 2, color=col, lw=2, solid_capstyle="round")
            ax.plot(r * 100, y + off, "o", ms=8, color=col, mec="white", mew=1.5, label=lab if i == 0 else None)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(list(reversed(rows)), fontsize=10, color=INK)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.axhline(0.5, color=GRID, lw=1)
    ax.set_xlim(0, 100)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xlabel("Posts and replies that leak owner information (%)", fontsize=9.5, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inviting")
    ap.add_argument("neutral")
    ap.add_argument("--out", default="results/bandwagon_pilot")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inv, neu = load(a.inviting), load(a.neutral)
    lines = ["topic,model,judge,env,episodes,writes,leaky_writes,rate,ci_lo,ci_hi"]
    for field, tag in [("judge_domain", "domain"), ("judge", "strict")]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
        jn = "domain judge: any owner detail" if tag == "domain" else "strict judge: exact attribute value"
        for ax, eps, envs, topic, t in [(axes[0], inv, ("TC", "TD"), "inviting",
                                         f"Inviting threads (\"how's your human?\"), {len(inv)} episodes"),
                                        (axes[1], neu, ("NC", "ND"), "neutral",
                                         f"Neutral threads (\"favorite debugging trick?\"), {len(neu)} episodes")]:
            for r in panel(ax, eps, envs, t, field):
                lines.append(f"{topic}," + ",".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r))
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, -0.28), ncol=2, frameon=False, fontsize=9.5)
        fig.suptitle(f"Peer oversharing barely moves leakage in the pilots ({jn}, no mitigation)",
                     fontsize=12.5, color=INK, x=0.01, ha="left", fontweight="bold")
        fig.text(0.01, 0.005, "Bars: 95% Wilson intervals over writes (optimistic: writes cluster within episodes). "
                 "Pilot 4 = inviting, pilot 5 = neutral; different model sets.", fontsize=8.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
        fig.savefig(out / f"bandwagon_{tag}.png", dpi=160, facecolor="white")
        plt.close(fig)
    (out / "bandwagon_pilot.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
