"""Calibration (reliability) and probability distributions of the final stack (5-fold OOF on val) -> images/calibration.png"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "src"); from common import load, LABELS
_, val, _ = load(); y = val.label.map(LABELS.index).values; P = np.load("preds/stack_oof.npy"); assert np.allclose(P.sum(1), 1, atol=1e-4) and len(P) == len(y)
pred = P.argmax(1); conf = P.max(1); ok = pred == y; B = np.linspace(0, 1, 11)
print(f"accuracy {ok.mean():.4f} | mean max-prob {conf.mean():.3f} | median conf correct {np.median(conf[ok]):.3f} wrong {np.median(conf[~ok]):.3f} | conf>0.9: {(conf > .9).mean():.3f} of rows, accuracy there {ok[conf > .9].mean():.4f}")
fig, ax = plt.subplots(2, 3, figsize=(13, 7)); col = ["#3b6fb6", "#3a9a5b", "#c0392b"]
for c, l in enumerate(LABELS):
    p, t = P[:, c], (y == c).astype(float); b = np.clip(np.digitize(p, B) - 1, 0, 9); m = np.array([[p[b == i].mean(), t[b == i].mean(), (b == i).sum()] for i in range(10) if (b == i).sum() >= 20])
    ece = sum(abs(r[0] - r[1]) * r[2] for r in m) / len(p); brier = ((p - t) ** 2).mean(); print(f"{l}: ECE {ece:.4f} | Brier {brier:.4f} | mean P {p.mean():.3f} vs true rate {t.mean():.3f}")
    a = ax[0, c]; a.plot([0, 1], [0, 1], "k--", lw=1); a.plot(m[:, 0], m[:, 1], "o-", color=col[c]); [a.text(r[0], r[1] + .03, f"{int(r[2])}", fontsize=6, ha="center") for r in m]
    a.set_xlabel(f"predicted P({l})"); a.set_ylabel("observed frequency"); a.set_title(f"calibration: {l} (ECE {ece:.3f})"); a.set_xlim(-.02, 1.02); a.set_ylim(-.02, 1.08)
    a = ax[1, c]; a.hist(p[y == c], bins=B, alpha=.8, color=col[c], label=f"true {l} (median {np.median(p[y == c]):.3f})"); a.hist(p[y != c], bins=B, alpha=.5, color="gray", label=f"other classes (median {np.median(p[y != c]):.3f})")
    a.set_yscale("log"); a.set_xlabel(f"predicted P({l})"); a.set_ylabel("rows (log)"); a.legend(fontsize=7); a.set_title(f"distribution of P({l})")
plt.tight_layout(); plt.savefig("images/calibration.png", dpi=150)
