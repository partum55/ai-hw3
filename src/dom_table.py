import sys, os; sys.path.insert(0, "src")
import numpy as np; from common import load, LABELS; from sklearn.metrics import f1_score
_, val, _ = load(); y = val.label.map(LABELS.index).values; dom = val.domain.values; DOMS = sorted(set(dom))
P = {"tfidf": "preds/tfidf_val.npy", "struct": "preds/struct_val.npy", "enc": "outputs/enc/enc_val.npy", "enc2": "outputs/enc2/enc2_val.npy", "enc3": "outputs/enc3/enc3_val.npy",
     "enc4": "preds/enc4_val.npy", "enc5": "outputs/enc5/enc5_val.npy", "enc6": "preds/enc6_val.npy", "encwin": "outputs/encwin/encwin_val.npy", "encwin7p": "preds/encwin7p_val.npy",
     "encwin8": "outputs/encwin8/encwin8_val.npy", "enc8": "preds/enc8_val.npy", "blogdeb": "outputs/blogdeb/blogdeb_val.npy", "blogrob": "outputs/blogrob/blogrob_val.npy",
     "qwen9": "preds/qwen9_val.npy", "STACK(14)": "preds/stack_oof.npy"}
pred = {k: np.load(p).argmax(1) for k, p in P.items() if os.path.exists(p)}
assert len(pred["STACK(14)"]) == len(y); print("stack all-rows F1", round(f1_score(y, pred["STACK(14)"], average="macro"), 4))
short = {"arxiv_papers": "arxiv", "blog_authorship_corpus": "blog", "bundestag": "bundest", "cnn_news": "cnn", "euro_court_cases": "eurocourt", "gutenberg": "gutenb", "house_of_commons": "commons", "student_essays": "essays"}
print("\nn per domain:", {short[d]: int((dom == d).sum()) for d in DOMS})
hdr = "| model | all | " + " | ".join(short[d] for d in DOMS) + " |"; sep = "|" + "---|" * (len(DOMS) + 2)
print("\n### macro-F1 per domain\n" + hdr + "\n" + sep)
for k, p in pred.items(): print(f"| {k} | {f1_score(y, p, average='macro'):.3f} | " + " | ".join(f"{f1_score(y[dom == d], p[dom == d], average='macro'):.3f}" for d in DOMS) + " |")
def conf(p, m):  # share of human+fusion rows of the mask that are swapped between human and fusion
    yy, pp = y[m], p[m]; hf = (yy != 1); return ((yy == 0) & (pp == 2)).sum() / max((yy == 0).sum(), 1), ((yy == 2) & (pp == 0)).sum() / max((yy == 2).sum(), 1)
print("\n### human->fusion % / fusion->human % (share of that class' rows, per domain)\n" + hdr + "\n" + sep)
for k, p in pred.items():
    c = [conf(p, np.ones(len(y), bool))] + [conf(p, dom == d) for d in DOMS]; print(f"| {k} | " + " | ".join(f"{100*a:.0f}/{100*b:.0f}" for a, b in c) + " |")
print("\n### AI-vs-rest check (ai recall %) all domains, to show ai is easy")
for k in ("tfidf", "struct", "enc3", "enc8", "qwen9", "STACK(14)"): print(k, round(100 * ((pred[k] == 1) & (y == 1)).sum() / (y == 1).sum(), 1))
print("\n### best / worst single model per domain (macro-F1, excl. STACK, enc8, blogdeb, blogrob outside own scope)")
for d in DOMS:
    m = dom == d; s = {k: f1_score(y[m], p[m], average="macro") for k, p in pred.items() if k not in ("STACK(14)", "enc8", "blogdeb", "blogrob")}
    b = max(s, key=s.get); w = min(s, key=s.get); print(f"{short[d]:9s} best {b} {s[b]:.3f} | worst {w} {s[w]:.3f} | stack {f1_score(y[m], pred['STACK(14)'][m], average='macro'):.3f}")
