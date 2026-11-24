"""Class-scale tuning for macro-F1 on the stack OOF probabilities (preds/stack_oof.npy). Honest check: tune on one half of val, score on the other."""
import itertools, numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
P, y = np.load("preds/stack_oof.npy"), np.load("preds/stack_oof_y.npy")
G = np.exp(np.linspace(-0.7, 0.7, 15))
def best(P, y):
    return max(((f1_score(y, (P * np.array([1, a, b])).argmax(1), average="macro"), a, b) for a, b in itertools.product(G, G)))
base = f1_score(y, P.argmax(1), average="macro"); gains = []
for k, (a, b) in enumerate(StratifiedKFold(2, shuffle=True, random_state=1).split(P, y)):
    _, sa, sb = best(P[a], y[a]); f = f1_score(y[b], (P[b] * np.array([1, sa, sb])).argmax(1), average="macro")
    gains.append(f - f1_score(y[b], P[b].argmax(1), average="macro")); print(f"half {k}: scales ai={sa:.2f} fusion={sb:.2f}  held-out gain {gains[-1]:+.4f}")
f, sa, sb = best(P, y); print(f"base {base:.4f} | mean held-out gain {np.mean(gains):+.4f} | full-val tuned {f:.4f} (optimistic) scales ai={sa:.2f} fusion={sb:.2f}")
