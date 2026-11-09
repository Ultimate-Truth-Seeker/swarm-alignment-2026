"""Paper figures from results/results.json. Usage: python3 figs.py results"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
R = json.load(open(os.path.join(OUT, "results.json")))
ACC, ORG, GREY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#9a9893", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "axes.labelcolor": INK})

# ---------- Figure 1: forest plot of LLM monitor views ----------
NAMES = {"dm_only": "DMs only", "random_0.25": "Random 25% of messages", "random_0.5": "Random 50% of messages",
         "full_bids": "Full log + bid table", "agent_2": "Tap 2 random agents", "adaptive_2": "Tap 2 agents (adaptive)",
         "full": "Full log", "channel_0.25": "Tap 25% of channels", "agent_1": "Tap 1 random agent",
         "bids_only": "Bid table only (no messages)", "channel_0.5": "Tap 50% of channels",
         "public_only": "Public board only"}
rows = [(v, r) for v, r in R["llm_monitor"].items() if "pooled" in r]
rows.sort(key=lambda t: t[1]["pooled"]["auroc"])
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=220)
for y, (v, r) in enumerate(rows):
    a, (lo, hi) = r["pooled"]["auroc"], r["pooled"]["ci"]
    c = ACC if v == "dm_only" else (ORG if v == "public_only" else GREY)
    ax.hlines(y, lo, hi, color=c, lw=1.6, alpha=0.9)
    ax.plot(a, y, "o", ms=6.5, color=c, mec="white", mew=1.2, zorder=3)
    ax.text(1.04, y, "%.2f" % a, va="center", fontsize=8, color=INK)
    ax.text(1.15, y, "%d%%" % round(100 * r["frac_obs"]), va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(rows)), [NAMES.get(v, v) for v, _ in rows])
ax.text(1.04, len(rows) - 0.3, "AUC", fontsize=7.5, color=INK2)
ax.text(1.15, len(rows) - 0.3, "% seen", fontsize=7.5, color=INK2)
refs = [("cover_bid_rate", "cover-bid rate", "--"), ("mean_winning_price", "mean price", ":")]
for j, (key, lab, ls) in enumerate(refs):
    v = R["baselines"][key]["pooled"]["auroc"]
    ax.axvline(v, color=INK2, ls=ls, lw=0.9)
    right = key == "cover_bid_rate"
    ax.text(v + (0.006 if right else -0.006), -1.0, lab + " %.2f" % v, ha="left" if right else "right",
            fontsize=7, color=INK2)
ax.axvline(0.5, color=GRID, lw=1)
ax.set_xlim(0.3, 1.015)
ax.set_ylim(-1.4, len(rows) - 0.4)
ax.set_xlabel("Episode-level AUROC, collusion vs clean (95%% bootstrap CI, n = %d episodes)" % R["n_episodes"])
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.suptitle("Where the monitor looks matters more than how much it sees", x=0.02, ha="left",
             fontsize=10.5, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 0.86, 1))
fig.savefig(os.path.join(OUT, "fig1_views_forest.png"))

# ---------- Figure 2: main vs metadata-aware colluders ----------
A = R["aware"]
pairs = [
    ("DM count (metadata)", R["baselines"]["dm_volume_metadata"]["pooled"]["auroc"], A["dm_volume"]["auroc"], A["dm_volume"]["ci"]),
    ("LLM monitor, full log", R["llm_monitor"]["full"]["pooled"]["auroc"], A["llm"]["full"]["auroc"], A["llm"]["full"]["ci"]),
    ("LLM monitor, random 25%", R["llm_monitor"]["random_0.25"]["pooled"]["auroc"], A["llm"]["random_0.25"]["auroc"], A["llm"]["random_0.25"]["ci"]),
    ("LLM monitor, public board", R["llm_monitor"]["public_only"]["pooled"]["auroc"], A["llm"]["public_only"]["auroc"], A["llm"]["public_only"]["ci"]),
    ("LLM monitor, bid table", R["llm_monitor"]["bids_only"]["pooled"]["auroc"], A["llm"]["bids_only"]["auroc"], A["llm"]["bids_only"]["ci"]),
    ("Lexicon, held-out terms", R["lexicon_fixed_views"]["full|heldout"]["auroc"], A["lexicon_heldout_full"]["auroc"], A["lexicon_heldout_full"]["ci"]),
    ("Mean winning price", R["baselines"]["mean_winning_price"]["pooled"]["auroc"], A["mean_price"]["auroc"], A["mean_price"]["ci"]),
    ("Cover-bid rate (bids)", R["baselines"]["cover_bid_rate"]["pooled"]["auroc"], A["cover_bid_rate"]["auroc"], A["cover_bid_rate"]["ci"]),
]
pairs = pairs[::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.3), dpi=220)
for y, (lab, m, a, ci) in enumerate(pairs):
    ax.hlines(y - 0.14, ci[0], ci[1], color=ACC, lw=1.1, alpha=0.5)
    ax.annotate("", xy=(a, y - 0.14), xytext=(m, y + 0.14),
                arrowprops=dict(arrowstyle="-", color=GRID, lw=1.2))
    ax.plot(m, y + 0.14, "o", ms=6.5, color=GREY, mec="white", mew=1.2, zorder=3)
    ax.plot(a, y - 0.14, "o", ms=6.5, color=ACC, mec="white", mew=1.2, zorder=3)
    ax.text(1.015, y, "%.2f → %.2f" % (m, a), va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(pairs)), [p[0] for p in pairs])
ax.axvline(0.5, color=INK2, ls=":", lw=0.9)
ax.text(0.5, len(pairs) - 0.45, "chance", ha="center", fontsize=7, color=INK2)
ax.set_xlim(0.2, 1.03)
ax.set_ylim(-0.6, len(pairs) - 0.3)
ax.set_xlabel("AUROC vs the 12 clean markets")
ax.plot([], [], "o", color=GREY, label="instructed colluders (n = 12)")
ax.plot([], [], "o", color=ACC, label="metadata-aware colluders (n = %d, 95%% CI)" % A["n_aware"])
ax.legend(loc="lower left", fontsize=7.5, frameon=False, ncol=2, bbox_to_anchor=(-0.45, 1.0))
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.suptitle("Colluders who avoid DMs evade message monitors, not their own bids", x=0.02, ha="left",
             fontsize=10.5, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0, 0.86, 0.97))
fig.savefig(os.path.join(OUT, "fig2_aware.png"))
print("ok")
