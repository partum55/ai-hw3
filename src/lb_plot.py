"""Public leaderboard score distribution -> images/lb_scores.png. Usage: python src/lb_plot.py <leaderboard.csv> [our_team_id]"""
import sys, numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
d = pd.read_csv(sys.argv[1], encoding="utf-8-sig"); me = int(sys.argv[2]) if len(sys.argv) > 2 else 17377056
s = d.Score.values; mine = d[d.TeamId == me].iloc[0]; q = np.percentile(s, [25, 50, 75])
print(f"teams {len(s)} | median {q[1]:.4f} | Q1 {q[0]:.4f} Q3 {q[2]:.4f} | min {s.min():.4f} max {s.max():.4f} | ours rank {int(mine.Rank)} score {mine.Score:.5f}")
print("top-10:", d.Score.head(10).round(5).tolist()); print("scores > 0.97:", int((s > .97).sum()), "| > 0.95:", int((s > .95).sum()), "| < 0.90:", int((s < .90).sum()))
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.2, 1.2, 1.4]})
ax[0].hist(s, bins=25, color="#9fb7d9", edgecolor="white"); ax[0].axvline(q[1], color="k", ls="--", label=f"median {q[1]:.3f}"); ax[0].axvline(mine.Score, color="#c0392b", lw=2, label=f"us {mine.Score:.4f}")
ax[0].set_xlabel("public macro-F1"); ax[0].set_ylabel("teams"); ax[0].legend(fontsize=8); ax[0].set_title(f"all {len(s)} teams")
ax[1].boxplot(s, vert=False, widths=.5, flierprops=dict(markersize=3)); ax[1].scatter([mine.Score], [1], color="#c0392b", zorder=3, s=40, label="us"); ax[1].set_yticks([]); ax[1].set_xlabel("public macro-F1"); ax[1].legend(fontsize=8); ax[1].set_title("box: Q1 / median / Q3")
t = d.head(25); ax[2].barh(t.Rank, t.Score, color=["#c0392b" if i == me else "#9fb7d9" for i in t.TeamId]); ax[2].invert_yaxis(); ax[2].set_xlim(t.Score.min() - .003, t.Score.max() + .002); ax[2].set_xlabel("public macro-F1"); ax[2].set_ylabel("rank"); ax[2].set_title("top 25 (zoom)")
plt.tight_layout(); plt.savefig("images/lb_scores.png", dpi=150)
