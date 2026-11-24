"""Long-text encoder: trains on a random <=510-token window of each text, predicts over ALL windows (stride 384, <=6 per text).
Usage: python src/enc_win.py <mdeberta|xlmr> <n_extra> <seed> <tag> [--smoke] [--infer [--weights=<old tag>] [--maxw=N] [--stride=N]]
Writes preds/encwin<tag>_{val,test}.npy (mean window probs, order human/ai/fusion), preds/encwin<tag>_x_{val,test}.npy
(max prob x3, std of prob x3, n_windows = window disagreement, the fusion 'patchwork' signal) and models/encwin<tag>.pt (best weights); preds/encwin<tag>_pos_*.npy = [max(3), first(3), last(3), middle-mean(3), n] per-window positional features."""
import sys, os, numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
sys.path.insert(0, "src"); from common import load, LABELS; from winutil import window_starts, aggregate, positional

NAMES = {"mdeberta": "microsoft/mdeberta-v3-base", "xlmr": "xlm-roberta-base", "deberta-v3": "microsoft/deberta-v3-base"}  # deberta-v3 is English-only: use with --en-only
name, N_EXTRA, SEED, TAG = NAMES[sys.argv[1]], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; SMOKE = "--smoke" in sys.argv; EN = "--en-only" in sys.argv
W, EPOCHS, BS, LR = 510, 2, 8, 2e-5
INFER = "--infer" in sys.argv  # skip training: load models/encwin<TAG>.pt and only predict val/test
arg = lambda k, d: next((type(d)(a.split("=")[1]) for a in sys.argv if a.startswith(k + "=")), d)
MAXW, STRIDE, WTAG = arg("--maxw", 6), arg("--stride", 384), arg("--weights", TAG)  # inference windows; WTAG = tag of the saved weights (a new TAG may reuse them)
INF_BS = 16  # inference batch: 64 x 512 tokens overflowed the 16 GB card for mDeBERTa (spilled to system RAM, ~4 windows/s)
fit, val, te = load()
if EN: fit = fit[fit.lang == "en-EN"]  # English-only training rows (val/test are still predicted in full; the stacker knows `lang`)
if N_EXTRA:  # extra training rows from the unused train fold, never val
    tr = pd.read_csv("data/raw/train.csv"); use = pd.read_csv("data/split.csv").use.values; pool = use == "unused"
    if EN: pool &= (tr.lang == "en-EN").values
    fit = pd.concat([fit, tr[pool].sample(N_EXTRA, random_state=SEED)])
if SMOKE: fit, val, te = fit.sample(300, random_state=0), val.sample(100, random_state=0), te.head(100)
hold = fit.sample(frac=0.1, random_state=0); fit = fit.drop(hold.index)  # epoch selection on a holdout of fit, never val
print(name, "train", len(fit), "holdout", len(hold), "val", len(val), "test", len(te), flush=True)

tok = AutoTokenizer.from_pretrained(name); CLS, SEP, PAD = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
def enc(df, step=2000):  # whole text, no truncation; int32 arrays keep 150k long texts in RAM
    return [np.asarray(x, np.int32) for i in range(0, len(df), step) for x in tok(list(df.text.iloc[i:i + step]), add_special_tokens=False).input_ids]
y = lambda df: torch.tensor(df.label.map(LABELS.index).values)
if INFER: Xv, Xt = enc(val), enc(te); Xf = Xh = []
else: Xf, Xh, Xv, Xt = enc(fit), enc(hold), enc(val), enc(te); print("tokenized; mean tokens", np.mean([len(x) for x in Xf]), flush=True)
def batch(segs):
    L = max(len(s) for s in segs) + 2; ids = torch.full((len(segs), L), PAD); m = torch.zeros_like(ids)
    for j, s in enumerate(segs): ids[j, 0], ids[j, 1:len(s) + 1], ids[j, len(s) + 1], m[j, :len(s) + 2] = CLS, torch.from_numpy(s.astype(np.int64)), SEP, 1
    return ids.cuda(), m.cuda()

model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3, dtype=torch.float32).cuda()
yf = y(fit); w = torch.tensor(len(yf) / (3 * np.bincount(yf.numpy(), minlength=3)), dtype=torch.float).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * (len(Xf) // BS); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
def predict(X, maxw=None):
    maxw = maxw or MAXW
    items = [(r, X[r][s:s + W]) for r in range(len(X)) for s in window_starts(len(X[r]), W, STRIDE, maxw)]
    order = np.argsort([len(t) for _, t in items]); probs = np.zeros((len(items), 3), np.float32); model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, len(items), INF_BS):
            idx = order[b:b + INF_BS]; ids, m = batch([items[j][1] for j in idx])
            probs[idx] = torch.softmax(model(input_ids=ids, attention_mask=m).logits.float(), -1).cpu().numpy()
            if (b // INF_BS) % 800 == 0: print(f"predict {b}/{len(items)} windows", flush=True)
    model.train(); torch.cuda.empty_cache()
    own = np.array([r for r, _ in items]); return (*aggregate(probs, own, len(X)), positional(probs, own, len(X)))

rng = np.random.RandomState(SEED); best, state = -1, None
if INFER: state = torch.load(f"models/encwin{WTAG}.pt")
for ep in range(0 if INFER else EPOCHS):
    perm = rng.permutation(len(Xf))
    for s in range(0, len(perm) - BS + 1, BS):
        idx = perm[s:s + BS]; segs = []
        for i in idx:  # random window of the text: every part of a long text gets trained on
            ids = Xf[i]; o = rng.randint(0, len(ids) - W + 1) if len(ids) > W else 0; segs.append(ids[o:o + W])
        ids, m = batch(segs)
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(input_ids=ids, attention_mask=m).logits
        loss = Fn.cross_entropy(logits.float(), yf[idx].cuda(), weight=w); assert torch.isfinite(loss), "non-finite loss"
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if (s // BS) % (10 if SMOKE else 500) == 0: print(f"ep {ep} {s}/{len(perm)} loss {loss.item():.4f}", flush=True)
    f1 = f1_score(y(hold).numpy(), predict(Xh, 3)[0].argmax(1), average="macro"); print("epoch", ep, "holdout macro-F1 (<=3 windows)", round(f1, 4), flush=True)
    if f1 > best: best, state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
if not INFER: os.makedirs("models", exist_ok=True); torch.save(state, f"models/encwin{TAG}.pt")  # weights on disk BEFORE inference (enc4 lesson)
model.load_state_dict(state); del opt, sch; torch.cuda.empty_cache()
for nm, X in (("val", Xv), ("test", Xt)):
    mean, ext, pos = predict(X); np.save(f"preds/encwin{TAG}_{nm}.npy", mean); np.save(f"preds/encwin{TAG}_x_{nm}.npy", ext); np.save(f"preds/encwin{TAG}_pos_{nm}.npy", pos)
    if nm == "val": print("VAL macro-F1 (all windows, mean prob)", round(f1_score(y(val).numpy(), mean.argmax(1), average="macro"), 4), flush=True)
print("saved", TAG)
