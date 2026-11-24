"""Windowed decoder-LLM classifier (Qwen2.5-0.5B, open weights): trains on a random <=510-token window of each text, predicts over ALL windows
(stride 384, <=6 per text) -> mean prob + window-max/std/n + positional features. Same recipe as src/enc_win.py, backbone/memory handling as src/llm_cls.py.
Usage: python src/llm_win.py <n_extra> <seed> <tag> [--smoke]
Writes preds/qwenw<tag>_{val,test}.npy (mean window probs, human/ai/fusion), _x_ (7 cols: max3, std3, n), _pos_ (13 cols), models/qwenw<tag>.pt (saved before inference)."""
import sys, os, numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
sys.path.insert(0, "src"); from common import load, LABELS; from winutil import window_starts, aggregate, positional

NAME = "Qwen/Qwen2.5-0.5B"
N_EXTRA, SEED, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; SMOKE = "--smoke" in sys.argv
W, STRIDE, MAXW, EPOCHS, BS, LR, INF_BS = 510, 384, 6, 2, 8, 1e-5, 16
fit, val, te = load()
if N_EXTRA:  # extra training rows from the unused train fold, never val
    tr = pd.read_csv("data/raw/train.csv"); use = pd.read_csv("data/split.csv").use.values
    fit = pd.concat([fit, tr[use == "unused"].sample(N_EXTRA, random_state=SEED)])
if SMOKE: fit, val, te = fit.sample(300, random_state=0), val.sample(100, random_state=0), te.head(100)
hold = fit.sample(frac=0.1, random_state=0); fit = fit.drop(hold.index)  # epoch selection on a holdout of fit, never val
print("train", len(fit), "holdout", len(hold), "val", len(val), "test", len(te), flush=True)

tok = AutoTokenizer.from_pretrained(NAME); model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=3, dtype=torch.float32)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
def enc(df, step=2000):  # whole text, no truncation, no special tokens (the head reads the last real token; right padding)
    return [np.asarray(x, np.int32) for i in range(0, len(df), step) for x in tok(list(df.text.iloc[i:i + step]), add_special_tokens=False).input_ids]
y = lambda df: torch.tensor(df.label.map(LABELS.index).values)
Xf, Xh, Xv, Xt = enc(fit), enc(hold), enc(val), enc(te); print("tokenized; mean tokens", np.mean([len(x) for x in Xf]), flush=True)
def batch(segs):
    L = max(len(s) for s in segs); ids = torch.full((len(segs), L), PAD); m = torch.zeros_like(ids)
    for j, s in enumerate(segs): ids[j, :len(s)] = torch.from_numpy(s.astype(np.int64)); m[j, :len(s)] = 1
    return ids.cuda(), m.cuda()

model = model.cuda()
for c in (model.config, getattr(model.config, "text_config", model.config)): c.pad_token_id = PAD; c.use_cache = False
model.gradient_checkpointing_enable(); model.get_input_embeddings().weight.requires_grad_(False)  # no grads / Adam state for the embedding matrix
yf = y(fit); w = torch.tensor(len(yf) / (3 * np.bincount(yf.numpy(), minlength=3)), dtype=torch.float).cuda()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
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
for ep in range(EPOCHS):
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
os.makedirs("models", exist_ok=True); torch.save(state, f"models/qwenw{TAG}.pt")  # weights on disk BEFORE inference (enc4 lesson)
model.load_state_dict(state); del opt, sch; torch.cuda.empty_cache()
for nm, X in (("val", Xv), ("test", Xt)):
    mean, ext, pos = predict(X); np.save(f"preds/qwenw{TAG}_{nm}.npy", mean); np.save(f"preds/qwenw{TAG}_x_{nm}.npy", ext); np.save(f"preds/qwenw{TAG}_pos_{nm}.npy", pos)
    if nm == "val": print("VAL macro-F1 (all windows, mean prob)", round(f1_score(y(val).numpy(), mean.argmax(1), average="macro"), 4), flush=True)
print("saved", TAG)
