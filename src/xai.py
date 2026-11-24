"""xAI for the structure group: TreeSHAP (LightGBM pred_contrib) on val. Same config as struct_model.py (with ppl features).
Prints per class: top features by mean |SHAP|, direction (corr of feature value with SHAP), class means of the feature -> images/xai_struct.png"""
import sys, numpy as np, pandas as pd, lightgbm as lgb, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src"); from common import load, LABELS

fit, val, _ = load()
F = pd.read_parquet("data/feats_struct.parquet").merge(pd.read_parquet("outputs/ppl/ppl_feats.parquet"), on="id").set_index("id")
cols = list(F.columns); y = lambda d: d.label.map(LABELS.index).values
hold = fit.sample(frac=0.1, random_state=0); tr_ = fit.drop(hold.index)
m = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                       class_weight="balanced", n_jobs=8, verbose=-1)
m.fit(F.loc[tr_.id, cols].values, y(tr_), eval_set=[(F.loc[hold.id, cols].values, y(hold))], callbacks=[lgb.early_stopping(50, verbose=False)])
X = F.loc[val.id, cols]; yv = y(val)
C = m.booster_.predict(X.values, pred_contrib=True).reshape(len(val), 3, len(cols) + 1)[:, :, :-1]  # [n, class, feature]
imp = np.abs(C).mean(0)  # [class, feature]
order = imp.sum(0).argsort()[::-1][:15]
fig, ax = plt.subplots(figsize=(7, 6)); left = np.zeros(15)
for c, lab in enumerate(LABELS): ax.barh(range(15), imp[c, order], left=left, label=lab); left += imp[c, order]
ax.set_yticks(range(15)); ax.set_yticklabels([cols[i] for i in order]); ax.invert_yaxis(); ax.legend(); ax.set_title("mean |SHAP| by class, top 15 (structure + ppl, val)")
plt.tight_layout(); plt.savefig("images/xai_struct.png", dpi=120)
means = X.groupby(val.label.values).mean()
for c, lab in enumerate(LABELS):
    print(f"\n== {lab}: top-8 features by mean|SHAP| | corr(value, SHAP) | class means human/ai/fusion")
    for i in imp[c].argsort()[::-1][:8]:
        r = np.corrcoef(X.iloc[:, i].fillna(X.iloc[:, i].median()), C[:, c, i])[0, 1]
        print(f"{cols[i]:22s} {imp[c, i]:.3f} | {r:+.2f} | {means.loc['human', cols[i]]:.2f} / {means.loc['ai', cols[i]]:.2f} / {means.loc['fusion', cols[i]]:.2f}")
