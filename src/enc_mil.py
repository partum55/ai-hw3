"""Multiple-instance (MIL) windowed encoder (mDeBERTa-v3-base). The label belongs to the WHOLE text, not to a window: each training text contributes up to K windows,
their 3-class logits are pooled as [mean, max] over windows and mapped by a small learned linear head (init = plain mean pooling) -> text-level cross-entropy.
So the model learns by itself which window carries the AI signal (fusion = rewritten passage) instead of being trained on noisy window labels as in src/enc_win.py.
Usage: python src/enc_mil.py <n_extra> <seed> <tag> [--smoke]
Writes preds/encmil<tag>_{val,test}.npy (text-level probs from the learned pooling; order human/ai/fusion), encmil<tag>w_* (mean of window probs),
encmil<tag>_x_* (max3, std3, n of window probs -> stack --winx features), encmil<tag>_pos_* (13 positional cols), models/encmil<tag>.pt (saved before inference)."""
import sys, os, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
sys.path.insert(0, "src"); from common import load, LABELS; from winutil import window_starts, aggregate, positional

name = "microsoft/mdeberta-v3-base"
N_EXTRA, SEED, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; SMOKE = "--smoke" in sys.argv
W, STRIDE, MAXW, K, EPOCHS, M, ACC, LR, HLR, INF_BS = 510, 384, 6, 4, 2, 2, 4, 2e-5, 1e-3, 16  # K windows/text in training, M texts per micro-batch (<= 8 windows), ACC micro-batches per step
fit, val, te = load()
if N_EXTRA:  # extra training rows from the unused train fold, never val
    tr = pd.read_csv("data/raw/train.csv"); use = pd.read_csv("data/split.csv").use.values
    fit = pd.concat([fit, tr[use == "unused"].sample(N_EXTRA, random_state=SEED)])
if SMOKE: fit, val, te = fit.sample(300, random_state=0), val.sample(100, random_state=0), te.head(100)
hold = fit.sample(frac=0.1, random_state=0); fit = fit.drop(hold.index)  # epoch selection on a holdout of fit, never val
print(name, "train", len(fit), "holdout", len(hold), "val", len(val), "test", len(te), flush=True)

tok = AutoTokenizer.from_pretrained(name); CLS, SEP, PAD = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
def enc(df, step=2000):  # whole text, no truncation; int32 arrays keep 150k long texts in RAM
    return [np.asarray(x, np.int32) for i in range(0, len(df), step) for x in tok(list(df.text.iloc[i:i + step]), add_special_tokens=False).input_ids]
y = lambda df: torch.tensor(df.label.map(LABELS.index).values)
Xf, Xh, Xv, Xt = enc(fit), enc(hold), enc(val), enc(te); print("tokenized; mean tokens", np.mean([len(x) for x in Xf]), flush=True)
def batch(segs):
    L = max(len(s) for s in segs) + 2; ids = torch.full((len(segs), L), PAD); m = torch.zeros_like(ids)
    for j, s in enumerate(segs): ids[j, 0], ids[j, 1:len(s) + 1], ids[j, len(s) + 1], m[j, :len(s) + 2] = CLS, torch.from_numpy(s.astype(np.int64)), SEP, 1
    return ids.cuda(), m.cuda()

model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3, dtype=torch.float32).cuda()
head = nn.Linear(6, 3).cuda()
with torch.no_grad(): head.weight.zero_(); head.weight[:, :3] = torch.eye(3); head.bias.zero_()  # start = mean pooling of window logits; the max part is learned
def pool(lg, owner, n):  # lg [windows,3] logits, owner [windows] = text index in 0..n-1 -> [n,3] text logits
    mean = torch.zeros(n, 3, device=lg.device).index_add(0, owner, lg) / torch.bincount(owner, minlength=n).clamp(min=1)[:, None]
    mx = torch.full((n, 3), -1e4, device=lg.device).scatter_reduce(0, owner[:, None].expand(-1, 3), lg, "amax", include_self=False)
    return head(torch.cat([mean, mx], 1))
def pick(ids, rng):  # up to K of the text's windows (random subset when there are more)
    st = list(window_starts(len(ids), W, STRIDE, MAXW))
    if len(st) > K: st = [st[i] for i in sorted(rng.choice(len(st), K, replace=False))]
    return [ids[s:s + W] for s in st]
yf = y(fit); w = torch.tensor(len(yf) / (3 * np.bincount(yf.numpy(), minlength=3)), dtype=torch.float).cuda()
opt = torch.optim.AdamW([{"params": model.parameters(), "lr": LR}, {"params": head.parameters(), "lr": HLR}], weight_decay=0.01)
steps = EPOCHS * (len(Xf) // (M * ACC)); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
def predict(X, maxw=None):
    maxw = maxw or MAXW
    items = [(r, X[r][s:s + W]) for r in range(len(X)) for s in window_starts(len(X[r]), W, STRIDE, maxw)]
    order = np.argsort([len(t) for _, t in items]); lgs = np.zeros((len(items), 3), np.float32); model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, len(items), INF_BS):
            idx = order[b:b + INF_BS]; ids, m = batch([items[j][1] for j in idx])
            lgs[idx] = model(input_ids=ids, attention_mask=m).logits.float().cpu().numpy()
            if (b // INF_BS) % 800 == 0: print(f"predict {b}/{len(items)} windows", flush=True)
    model.train(); torch.cuda.empty_cache()
    own = np.array([r for r, _ in items])
    with torch.no_grad(): text = torch.softmax(pool(torch.from_numpy(lgs).cuda(), torch.from_numpy(own).cuda(), len(X)), -1).cpu().numpy()
    e = np.exp(lgs - lgs.max(1, keepdims=True)); probs = e / e.sum(1, keepdims=True)
    return (text, *aggregate(probs, own, len(X)), positional(probs, own, len(X)))  # text-level probs, window mean, window extras, positional

rng = np.random.RandomState(SEED); best, state = -1, None; model.train(); opt.zero_grad()
for ep in range(EPOCHS):
    perm = rng.permutation(len(Xf)); micro = 0
    for s in range(0, len(perm) - M + 1, M):
        idx = perm[s:s + M]; segs, owner = [], []
        for j, i in enumerate(idx): ws = pick(Xf[i], rng); segs += ws; owner += [j] * len(ws)
        ids, m = batch(segs)
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(input_ids=ids, attention_mask=m).logits
        out = pool(lg.float(), torch.tensor(owner).cuda(), len(idx))
        loss = Fn.cross_entropy(out, yf[idx].cuda(), weight=w) / ACC; assert torch.isfinite(loss), "non-finite loss"
        loss.backward(); micro += 1
        if micro % ACC == 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0); opt.step(); sch.step(); opt.zero_grad()
        if (micro % (10 if SMOKE else 1000)) == 0: print(f"ep {ep} {s}/{len(perm)} loss {loss.item() * ACC:.4f}", flush=True)
    f1 = f1_score(y(hold).numpy(), predict(Xh, 3)[0].argmax(1), average="macro"); print("epoch", ep, "holdout macro-F1 (<=3 windows, learned pooling)", round(f1, 4), flush=True)
    if f1 > best: best, state = f1, {"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
os.makedirs("models", exist_ok=True); torch.save(state, f"models/encmil{TAG}.pt")  # weights on disk BEFORE inference (enc4 lesson)
model.load_state_dict(state["model"]); head.load_state_dict(state["head"]); del opt, sch; torch.cuda.empty_cache()
for nm, X in (("val", Xv), ("test", Xt)):
    text, mean, ext, pos = predict(X)
    np.save(f"preds/encmil{TAG}_{nm}.npy", text); np.save(f"preds/encmil{TAG}w_{nm}.npy", mean); np.save(f"preds/encmil{TAG}_x_{nm}.npy", ext); np.save(f"preds/encmil{TAG}_pos_{nm}.npy", pos)
    if nm == "val": print("VAL macro-F1 text-level (learned pooling)", round(f1_score(y(val).numpy(), text.argmax(1), average="macro"), 4), "| window-mean", round(f1_score(y(val).numpy(), mean.argmax(1), average="macro"), 4), flush=True)
print("saved", TAG)
