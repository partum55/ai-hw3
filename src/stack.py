"""Stack class probabilities of base models. Meta-learner: 5-fold CV inside val (reported score), then fit on all val -> test."""
import sys, os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, classification_report, confusion_matrix
sys.path.insert(0, "src"); from common import load, LABELS

SRC = {"tfidf": "preds/tfidf_{}.npy", "struct": "preds/struct_{}.npy", "enc": "outputs/enc/enc_{}.npy",
       "enc2": "outputs/enc2/enc2_{}.npy", "enc3": "outputs/enc3/enc3_{}.npy", "enc4": "preds/enc4_{}.npy", "enc5": "outputs/enc5/enc5_{}.npy", "enc6": "preds/enc6_{}.npy", "pos": "preds/postfidf_{}.npy", "encwin": "outputs/encwin/encwin_{}.npy", "encwin7": "preds/encwin7_{}.npy", "encwin7p": "preds/encwin7p_{}.npy", "enc8": "preds/enc8_{}.npy", "qwen9": "preds/qwen9_{}.npy", "encmil12": "preds/encmil12_{}.npy", "encmil12w": "preds/encmil12w_{}.npy", "qwenw11": "preds/qwenw11_{}.npy", "encwin9": "outputs/encwin9/encwin9_{}.npy", "encwin10": "preds/encwin10_{}.npy", "encwin8": "outputs/encwin8/encwin8_{}.npy", "seq": "outputs/seq/seq_{}.npy", "blogdeb": "outputs/blogdeb/blogdeb_{}.npy", "blogrob": "outputs/blogrob/blogrob_{}.npy"}
use = [k for k in ([a for a in sys.argv[1:] if not a.startswith("--")] or SRC) if os.path.exists(SRC[k].format("val"))]
_, val, te = load(); y = val.label.map(LABELS.index).values
sample = pd.read_csv("data/raw/sample_submission.csv")
lg = lambda P: np.log(np.clip(P, 1e-6, 1))
V = {k: np.load(SRC[k].format("val")) for k in use}; T = {k: np.load(SRC[k].format("test")) for k in use}
assert all(v.shape == (len(val), 3) for v in V.values()) and all(t.shape == (len(te), 3) for t in T.values())
for k in use: print(f"{k:8s} alone: val macro-F1 {f1_score(y, V[k].argmax(1), average='macro'):.4f}")
Xv, Xt = np.hstack([lg(V[k]) for k in use]), np.hstack([lg(T[k]) for k in use])
DOM = sorted(set(val.domain) | set(te.domain))
ctx = lambda d: np.hstack([(d.lang == "de-DE").values[:, None], np.stack([(d.domain == x).values for x in DOM], 1)]).astype(float)
Cv, Ct = ctx(val), ctx(te)
XS = {"encwin": "outputs/encwin/encwin_x_{}.npy", "encwin7": "preds/encwin7_x_{}.npy", "encwin7p": "preds/encwin7p_x_{}.npy", "encwin8": "outputs/encwin8/encwin8_x_{}.npy", "qwenw11": "preds/qwenw11_x_{}.npy", "encmil12": "preds/encmil12_x_{}.npy", "encwin9": "outputs/encwin9/encwin9_x_{}.npy", "encwin10": "preds/encwin10_x_{}.npy"}  # window-disagreement features (max/std of window probs, n windows) -> meta context
wx = next((a for a in sys.argv if a.startswith("--winx")), None)  # --winx = all 7 columns, --winx=6 / --winx=3,4,5 = column subset (ablation)
if wx:
    cols = [int(c) for c in wx.split("=")[1].split(",")] if "=" in wx else list(range(7))
    for k in use:
        if k in XS: Cv, Ct = np.hstack([Cv, np.load(XS[k].format("val"))[:, cols]]), np.hstack([Ct, np.load(XS[k].format("test"))[:, cols]])
if "--pos" in sys.argv and "encwin7p" in use:  # positional window features (max/first/last/middle-mean/n per class, 13 cols) of the 8-window re-inference
    Cv, Ct = np.hstack([Cv, np.load("preds/encwin7p_pos_val.npy")]), np.hstack([Ct, np.load("preds/encwin7p_pos_test.npy")])
LR = lambda seed=0: LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
import lightgbm as lgb
LGB = lambda seed=0: lgb.LGBMClassifier(random_state=seed, n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=50, colsample_bytree=0.8,
                                 subsample=0.8, subsample_freq=1, class_weight="balanced", n_jobs=8, verbose=-1)
# meta variants: logreg on log-probs | + lang/domain one-hot | LightGBM on log-probs + lang/domain
META = {"lr": (LR, Xv, Xt), "lgb": (LGB, Xv, Xt), "lr+ctx": (LR, np.hstack([Xv, Cv]), np.hstack([Xt, Ct])), "lgb+ctx": (LGB, np.hstack([Xv, Cv]), np.hstack([Xt, Ct]))}
pick = next((a.split("=")[1] for a in sys.argv if a.startswith("--meta=")), "lr")
if "--check" in sys.argv:  # honesty check: meta fit on one random half of val, scored on the other (5 random splits x 2 directions); single seed vs 5-seed average
    mk, A, _ = META[pick]; r1, r5 = [], []
    for sd in range(5):
        a, b = next(StratifiedKFold(2, shuffle=True, random_state=100 + sd).split(A, y))
        for i, j in ((a, b), (b, a)):
            r1.append(f1_score(y[j], mk(0).fit(A[i], y[i]).predict_proba(A[j]).argmax(1), average="macro"))
            r5.append(f1_score(y[j], np.mean([mk(s).fit(A[i], y[i]).predict_proba(A[j]) for s in range(5)], 0).argmax(1), average="macro"))
    print(f"HALF-SPLIT meta={pick} (meta trained on 13.6k rows): single seed {np.mean(r1):.4f} +- {np.std(r1):.4f} | 5-seed avg {np.mean(r5):.4f} +- {np.std(r5):.4f} (10 evaluations)"); sys.exit()
cv, oofs = StratifiedKFold(5, shuffle=True, random_state=0), {}
only = any(a.startswith("--meta=") for a in sys.argv) and "--all" not in sys.argv  # with --meta=X compute only that meta (4x faster)
for k, (mk, A, _) in META.items():
    if only and k != pick: continue
    oofs[k] = cross_val_predict(mk(), A, y, cv=cv, method="predict_proba")
    print(f"STACK {use} meta={k:8s}: 5-fold CV on val macro-F1 {f1_score(y, oofs[k].argmax(1), average='macro'):.4f}")
np.save("preds/stack_oof.npy", oofs[pick]); np.save("preds/stack_oof_y.npy", y)  # for thresh.py
print(f"\nmeta={pick}"); print(classification_report(y, oofs[pick].argmax(1), target_names=LABELS, digits=3)); print(confusion_matrix(y, oofs[pick].argmax(1)))
print("per-domain macro-F1 (OOF):", {d: round(f1_score(y[(val.domain == d).values], oofs[pick].argmax(1)[(val.domain == d).values], average="macro"), 3) for d in DOM})
if "--write" in sys.argv:
    mk, A, At = META[pick]; pred = np.array(LABELS)[mk().fit(A, y).predict(At)]
    sub = pd.DataFrame({"id": te.id.values, "label": pred}).set_index("id").loc[sample.id].reset_index()
    assert len(sub) == 25000 and sub.id.is_unique and set(sub.label) <= set(LABELS)
    os.makedirs("submissions", exist_ok=True); fn = f"submissions/sub_stack_{pick.replace('+', '_')}.csv"; sub.to_csv(fn, index=False); print("wrote", fn)
