"""Report figures from our saved results -> images/report/*.png. Run: .venv/bin/python -W ignore -I src/make_figs.py
fig1_ladder (CV vs public per stack version), fig2_heatmap (model x domain F1), fig3_errors (confusion + human<->fusion by domain),
fig4_ablations (window features, head+tail vs head-only), fig5_models (single-model val F1 vs stack)."""
import sys, os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; from sklearn.metrics import f1_score, confusion_matrix
sys.path.insert(0, "src"); from common import load, LABELS
os.makedirs("images/report", exist_ok=True)
GROUP = {"tfidf": "tokens", "struct": "structure", "enc": "encoder", "enc2": "encoder", "enc3": "encoder", "enc4": "encoder", "enc5": "encoder", "enc6": "encoder",
         "encwin": "windowed", "encwin7p": "windowed", "encwin8": "windowed", "encwin10": "specialist", "enc8": "specialist", "blogdeb": "specialist", "blogrob": "specialist",
         "qwen9": "decoder LLM", "STACK": "stack"}
COL = {"tokens": "#8c8c8c", "structure": "#b0a090", "encoder": "#3b6fb6", "windowed": "#e08a1e", "specialist": "#3a9a5b", "decoder LLM": "#8e5bb5", "stack": "#c0392b"}
P = {"tfidf": "preds/tfidf_val.npy", "struct": "preds/struct_val.npy", "enc": "outputs/enc/enc_val.npy", "enc2": "outputs/enc2/enc2_val.npy", "enc3": "outputs/enc3/enc3_val.npy",
     "enc4": "preds/enc4_val.npy", "enc5": "outputs/enc5/enc5_val.npy", "enc6": "preds/enc6_val.npy", "encwin": "outputs/encwin/encwin_val.npy", "encwin7p": "preds/encwin7p_val.npy",
     "encwin8": "outputs/encwin8/encwin8_val.npy", "encwin10": "preds/encwin10_val.npy", "enc8": "preds/enc8_val.npy", "blogdeb": "outputs/blogdeb/blogdeb_val.npy",
     "qwen9": "preds/qwen9_val.npy", "STACK": "preds/stack_oof.npy"}
_, val, _ = load(); y = val.label.map(LABELS.index).values; dom = val.domain.values; DOMS = sorted(set(dom))
SHORT = {"arxiv_papers": "arxiv", "blog_authorship_corpus": "blog", "bundestag": "bundestag (DE)", "cnn_news": "cnn", "euro_court_cases": "euro court", "gutenberg": "gutenberg", "house_of_commons": "commons", "student_essays": "essays"}
pred = {k: np.load(p).argmax(1) for k, p in P.items()}
stack_f1 = f1_score(y, pred["STACK"], average="macro"); print("stack OOF macro-F1", round(stack_f1, 4), "(expected 0.9778 for the 15-model stack)")

# fig1: CV vs public per stack version (numbers from notes.md / Kaggle submissions)
V = [("baseline\nTF-IDF", 0.866, 0.8693), ("3 models\n+struct+enc", 0.9396, 0.9417), ("4 models\nLGB meta", 0.9546, 0.9515), ("6 models", 0.9591, 0.9569), ("9 models\n+window-max", 0.9703, 0.9684),
     ("10 models", 0.9736, 0.9717), ("12 models\n+blog+EN", 0.9767, 0.9755), ("15 models\n+Qwen,win", 0.9778, 0.9779)]
fig, ax = plt.subplots(figsize=(9, 4.2)); x = range(len(V))
ax.plot(x, [v[1] for v in V], "o-", color="#3b6fb6", label="val CV (5-fold in val, 27 236)"); ax.plot(x, [v[2] for v in V], "s--", color="#c0392b", label="Kaggle public (30 % of test)")
ax.set_xticks(list(x)); ax.set_xticklabels([v[0] for v in V], fontsize=8); ax.set_ylim(0.86, 0.985); ax.set_ylabel("macro-F1"); ax.grid(alpha=.3); ax.legend(loc="lower right")
ax.set_title("Stack versions: validation CV vs public leaderboard"); plt.tight_layout(); plt.savefig("images/report/fig1_ladder.png", dpi=150); plt.close()

# fig2: heatmap model x domain
names = list(pred); M = np.array([[f1_score(y[dom == d], pred[k][dom == d], average="macro") for d in DOMS] for k in names])
fig, ax = plt.subplots(figsize=(9, 6.2)); im = ax.imshow(M, vmin=0.5, vmax=1.0, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(len(DOMS))); ax.set_xticklabels([SHORT[d] for d in DOMS], rotation=30, ha="right", fontsize=8); ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
for i in range(len(names)):
    for j in range(len(DOMS)): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, label="macro-F1"); ax.set_title("macro-F1 per domain on val (German only in bundestag; enc8/blogdeb are English-only)"); plt.tight_layout(); plt.savefig("images/report/fig2_heatmap.png", dpi=150); plt.close()

# fig3: confusion of the final stack + human<->fusion error rates per domain
cm = confusion_matrix(y, pred["STACK"], normalize="true"); fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
a.imshow(cm, cmap="Blues", vmin=0, vmax=1); a.set_xticks(range(3)); a.set_xticklabels(LABELS); a.set_yticks(range(3)); a.set_yticklabels(LABELS); a.set_xlabel("predicted"); a.set_ylabel("true")
for i in range(3):
    for j in range(3): a.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black")
a.set_title("final stack, row-normalised")
hf = [((y[dom == d] == 0) & (pred["STACK"][dom == d] == 2)).sum() / max((y[dom == d] == 0).sum(), 1) * 100 for d in DOMS]
fh = [((y[dom == d] == 2) & (pred["STACK"][dom == d] == 0)).sum() / max((y[dom == d] == 2).sum(), 1) * 100 for d in DOMS]; w = .38
b.bar(np.arange(len(DOMS)) - w / 2, hf, w, label="human -> fusion", color="#3b6fb6"); b.bar(np.arange(len(DOMS)) + w / 2, fh, w, label="fusion -> human", color="#c0392b")
b.set_xticks(range(len(DOMS))); b.set_xticklabels([SHORT[d] for d in DOMS], rotation=30, ha="right", fontsize=8); b.set_ylabel("% of class rows"); b.legend(); b.set_title("final stack: human<->fusion errors by domain\n(gutenberg has only 20 human, 39 fusion rows)", fontsize=10)
plt.tight_layout(); plt.savefig("images/report/fig3_errors.png", dpi=150); plt.close()

# fig4: ablations (numbers from notes.md, same 5-fold CV in val)
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
A = [("encwin probs only\n(no extra cols)", 0.9663), ("n windows", 0.9664), ("std of window probs", 0.9665), ("max of window probs", 0.9703), ("max + std", 0.9703), ("all 7 columns", 0.9702)]
a.barh(range(len(A)), [v for _, v in A], color=["#8c8c8c"] * 3 + ["#e08a1e"] * 3); a.set_yticks(range(len(A))); a.set_yticklabels([n for n, _ in A], fontsize=8); a.invert_yaxis(); a.set_xlim(0.96, 0.972)
for i, (_, v) in enumerate(A): a.text(v + 0.0001, i, f"{v:.4f}", va="center", fontsize=8)
a.set_xlabel("stack CV macro-F1"); a.set_title("window-feature ablation (8-model stack + encwin)")
T = [("head 510\n(enc5)", 0.8733), ("head 128 + tail 382\n(enc)", 0.9248)]; b.bar(range(2), [v for _, v in T], color=["#8c8c8c", "#3b6fb6"]); b.set_xticks(range(2)); b.set_xticklabels([n for n, _ in T], fontsize=8); b.set_ylim(0.8, 0.95)
for i, (_, v) in enumerate(T): b.text(i, v + .002, f"{v:.4f}", ha="center", fontsize=9)
b.set_ylabel("single-model val macro-F1"); b.set_title("same mDeBERTa, same 50k rows: truncation choice"); plt.tight_layout(); plt.savefig("images/report/fig4_ablations.png", dpi=150); plt.close()

# fig5: single-model val F1 coloured by family vs stack
s = sorted(((f1_score(y, p, average="macro"), k) for k, p in pred.items() if k not in ("enc8", "blogdeb", "encwin10")), reverse=True)
fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(range(len(s)), [v for v, _ in s], color=[COL[GROUP[k]] for _, k in s]); ax.set_yticks(range(len(s))); ax.set_yticklabels([k for _, k in s], fontsize=8); ax.invert_yaxis(); ax.set_xlim(0.78, 1.0)
for i, (v, _) in enumerate(s): ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_xlabel("val macro-F1 (all rows; stack = 5-fold OOF)"); ax.set_title("single models vs stack (English-only specialists omitted: enc8, blogdeb, encwin10)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[g]) for g in ("tokens", "structure", "encoder", "windowed", "decoder LLM", "stack")], labels=["tokens", "structure", "encoder head+tail", "windowed", "decoder LLM", "stack"], loc="lower left", fontsize=8)
plt.tight_layout(); plt.savefig("images/report/fig5_models.png", dpi=150); plt.close(); print("saved images/report/*.png")
