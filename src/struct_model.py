"""LightGBM on structure features (+ optional ppl_feats). Early stopping only on a holdout of `fit`, never on val."""
import sys, os, numpy as np, pandas as pd, lightgbm as lgb, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, confusion_matrix
sys.path.insert(0, "src"); from common import load, LABELS

fit, val, te = load()
F = pd.read_parquet("data/feats_struct.parquet")
if os.path.exists("outputs/ppl/ppl_feats.parquet") and "--noppl" not in sys.argv:
    F = F.merge(pd.read_parquet("outputs/ppl/ppl_feats.parquet"), on="id", how="left"); print("with ppl features")
F = F.set_index("id")
y = lambda d: d.label.map(LABELS.index).values
hold = fit.sample(frac=0.1, random_state=0); tr_ = fit.drop(hold.index)

def run(cols, tag, save=False):
    X = lambda d: F.loc[d.id, cols].values
    m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                           class_weight="balanced", n_jobs=8, verbose=-1)
    m.fit(X(tr_), y(tr_), eval_set=[(X(hold), y(hold))], callbacks=[lgb.early_stopping(50, verbose=False)])
    P = m.predict_proba(X(val)); f1 = f1_score(y(val), P.argmax(1), average="macro")
    print(f"{tag:28s} n_feat={len(cols):4d} trees={m.best_iteration_:4d} val macro-F1 {f1:.4f}", flush=True)
    if save:
        np.save("preds/struct_val.npy", P); np.save("preds/struct_test.npy", m.predict_proba(X(te)))
        print(classification_report(y(val), P.argmax(1), target_names=LABELS, digits=3)); print(confusion_matrix(y(val), P.argmax(1)))
        imp = pd.Series(m.booster_.feature_importance("gain"), cols).sort_values()[-25:]
        imp.plot.barh(figsize=(7, 7)); plt.title("LightGBM gain, top 25"); plt.tight_layout(); plt.savefig("images/struct_importance.png", dpi=120); plt.close()
        print("top-10 gain:", ", ".join(imp.index[::-1][:10]))
    return f1

allc = [c for c in F.columns]
grp = lambda *p: [c for c in allc if c.startswith(p)]
run(grp("all_"), "whole-text regex only (no seg)")
run(grp("intro_"), "intro only"); run(grp("body_"), "body only"); run(grp("concl_", "concl_marker"), "conclusion only")
run(grp("x_", "concl_marker", "intro_ques", "intro_excl"), "cross-segment only")
run([c for c in allc if not c.startswith("all_")], "segments (no whole-text)")
run(allc, "ALL structure features", save=True)
