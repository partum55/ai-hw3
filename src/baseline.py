"""TF-IDF word+char baseline. Fixed split saved to data/split.csv (reused by every later model)."""
import os, sys, numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sys.path.insert(0, "src"); from common import load, LABELS
fit, val, te = load()
sample = pd.read_csv("data/raw/sample_submission.csv")

# vectorizers are fit on the train subsample only (no val/test leakage)
word = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=300_000, sublinear_tf=True)
char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=300_000, sublinear_tf=True)
feats = lambda d, f: hstack([word.transform(d.text) if not f else word.fit_transform(d.text),
                             char.transform(d.text) if not f else char.fit_transform(d.text)]).tocsr()
Xf = feats(fit, True); Xv = feats(val, False)

best = (-1, None, None)
for name, mk in {"logreg": lambda C: LogisticRegression(C=C, max_iter=300, class_weight="balanced"),
                 "linsvc": lambda C: LinearSVC(C=C, class_weight="balanced")}.items():
    for C in (0.3, 1, 3):
        m = mk(C).fit(Xf, fit.label); f1 = f1_score(val.label, m.predict(Xv), average="macro")
        print(f"{name} C={C}: val macro-F1 {f1:.4f}")
        if f1 > best[0]: best = (f1, f"{name} C={C}", m)
f1, name, m = best
pv = m.predict(Xv)
print(f"\nbest {name}: {f1:.4f}\n", classification_report(val.label, pv, digits=3))
labs = LABELS
print("confusion (rows=true, cols=pred)", labs, "\n", confusion_matrix(val.label, pv, labels=labs))

Xt = feats(te, False)
os.makedirs("preds", exist_ok=True)
for nm, X in (("val", Xv), ("test", Xt)):  # probabilities in LABELS order, for stacking
    if hasattr(m, "predict_proba"): np.save(f"preds/tfidf_{nm}.npy", m.predict_proba(X)[:, [list(m.classes_).index(l) for l in LABELS]])
pred = m.predict(Xt)
sub = pd.DataFrame({"id": te.id, "label": pred}).set_index("id").loc[sample.id].reset_index()
assert len(sub) == 25000 and sub.id.is_unique and set(sub.label) <= set(labs)
os.makedirs("submissions", exist_ok=True); sub.to_csv("submissions/sub_baseline.csv", index=False); print("wrote submissions/sub_baseline.csv")
