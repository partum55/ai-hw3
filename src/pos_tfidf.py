"""Positional bag-of-words: separate word TF-IDF vocabularies for intro / body / conclusion (first / last 20 % of sentences).
Control = the same word TF-IDF on the whole text (no position). Vectorizers fit on `fit` rows only. -> preds/postfidf_{val,test}.npy"""
import sys, numpy as np, pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
sys.path.insert(0, "src"); from common import load, segment, LABELS

fit, val, te = load()
def parts(df):  # texts with < 5 sentences have no segments -> whole text goes to the intro slot
    s = [segment(t) or (t, "", "") for t in df.text]
    return [[x[i] for x in s] for i in range(3)]
P = {n: parts(d) for n, d in (("fit", fit), ("val", val), ("test", te))}
mk = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
y = fit.label.map(LABELS.index).values; yv = val.label.map(LABELS.index).values
LR = lambda: LogisticRegression(C=3, max_iter=300, class_weight="balanced")

vec = [mk().fit(P["fit"][i]) for i in range(3)]
X = {n: hstack([vec[i].transform(P[n][i]) for i in range(3)]).tocsr() for n in P}
m = LR().fit(X["fit"], y); pv = m.predict_proba(X["val"]); f1p = f1_score(yv, pv.argmax(1), average="macro")
print(f"positional (3 x 200k word 1-2grams): val macro-F1 {f1p:.4f}", flush=True)
print(classification_report(yv, pv.argmax(1), target_names=LABELS, digits=3)); print(confusion_matrix(yv, pv.argmax(1)), flush=True)
np.save("preds/postfidf_val.npy", pv); np.save("preds/postfidf_test.npy", m.predict_proba(X["test"]))

# per-part strength: which segment's words carry the signal
for i, nm in enumerate(("intro", "body", "concl")):
    mi = LR().fit(vec[i].transform(P["fit"][i]), y)
    print(f"{nm:6s} words only: val macro-F1 {f1_score(yv, mi.predict(vec[i].transform(P['val'][i])), average='macro'):.4f}", flush=True)

# control: same kind of features, no position (whole text, 200k vocabulary)
vc = mk().fit(fit.text); mc = LR().fit(vc.transform(fit.text), y)
print(f"control whole-text words (no position): val macro-F1 {f1_score(yv, mc.predict(vc.transform(val.text)), average='macro'):.4f}")
