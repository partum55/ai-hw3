"""GPU (Kaggle T4): long-text encoder. Trains on a random <=510-token window of each text, predicts over ALL windows (stride 384, <=6).
Writes encwin_{val,test}.npy (mean window probs, order human/ai/fusion) and encwin_x_{val,test}.npy (max prob x3, std x3, n_windows)."""
import glob, numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
SMOKE = False  # 300 texts; set False for the full run
find = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
tr, te, sp = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv")), pd.read_csv(find("split.csv"))
assert (tr.id.values == sp.id.values).all()
tr["use"] = sp.use.values
LABELS = ["human", "ai", "fusion"]; name = "xlm-roberta-base"; N_EXTRA, SEED = 100_000, 4  # seed differs from enc2/enc3 (1, 2) and the friend-GPU runs (3, 5)
W, EPOCHS, BS, LR = 510, 2, 8, 2e-5
fit, val = tr[tr.use == "fit"], tr[tr.use == "val"]
fit = pd.concat([fit, tr[tr.use == "unused"].sample(N_EXTRA, random_state=SEED)])  # extra rows from the unused train fold, never val
if SMOKE: fit, val, te = fit.sample(300, random_state=0), val.sample(100, random_state=0), te.head(100)
hold = fit.sample(frac=0.1, random_state=0); fit = fit.drop(hold.index)  # epoch selection on a holdout of fit, never val

def window_starts(n, W=510, stride=384, maxw=6):  # same as src/winutil.py (self-checked there)
    if n <= W: return [0]
    s = list(range(0, n - W + 1, stride))
    if s[-1] != n - W: s.append(n - W)
    if len(s) > maxw: s = [s[round(k * (len(s) - 1) / (maxw - 1))] for k in range(maxw)]
    return s
def aggregate(probs, owner, n_texts):
    b = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]]); cnt = np.diff(np.r_[b, len(owner)]).astype(np.float32)
    mean = np.add.reduceat(probs, b) / cnt[:, None]
    std = np.sqrt(np.maximum(np.add.reduceat(probs ** 2, b) / cnt[:, None] - mean ** 2, 0))
    assert len(b) == n_texts
    return mean, np.hstack([np.maximum.reduceat(probs, b), std, cnt[:, None]])

tok = AutoTokenizer.from_pretrained(name); CLS, SEP, PAD = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id
def enc(df, step=2000):
    return [np.asarray(x, np.int32) for i in range(0, len(df), step) for x in tok(list(df.text.iloc[i:i + step]), add_special_tokens=False).input_ids]
y = lambda df: torch.tensor(df.label.map(LABELS.index).values)
Xf, Xh, Xv, Xt = enc(fit), enc(hold), enc(val), enc(te)
L = np.array([len(x) for x in Xf]); print("train", len(Xf), "mean tokens", L.mean(), "share > 510:", (L > W).mean(), flush=True)
def batch(segs):
    n = max(len(s) for s in segs) + 2; ids = torch.full((len(segs), n), PAD); m = torch.zeros_like(ids)
    for j, s in enumerate(segs): ids[j, 0], ids[j, 1:len(s) + 1], ids[j, len(s) + 1], m[j, :len(s) + 2] = CLS, torch.from_numpy(s.astype(np.int64)), SEP, 1
    return ids.cuda(), m.cuda()

model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3, torch_dtype=torch.float32).cuda()
yf = y(fit); w = torch.tensor(len(yf) / (3 * np.bincount(yf.numpy(), minlength=3)), dtype=torch.float).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * (len(Xf) // BS); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps); scaler = torch.amp.GradScaler()
def predict(X, maxw=6):
    items = [(r, X[r][s:s + W]) for r in range(len(X)) for s in window_starts(len(X[r]), W, 384, maxw)]
    order = np.argsort([len(t) for _, t in items]); probs = np.zeros((len(items), 3), np.float32); model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for b in range(0, len(items), 64):
            idx = order[b:b + 64]; ids, m = batch([items[j][1] for j in idx])
            probs[idx] = torch.softmax(model(input_ids=ids, attention_mask=m).logits.float(), -1).cpu().numpy()
    model.train(); torch.cuda.empty_cache()
    return aggregate(probs, np.array([r for r, _ in items]), len(X))

rng = np.random.RandomState(SEED); best, state = -1, None
for ep in range(EPOCHS):
    perm = rng.permutation(len(Xf))
    for s in range(0, len(perm) - BS + 1, BS):
        idx = perm[s:s + BS]; segs = []
        for i in idx:  # random window: every part of a long text gets trained on
            ids = Xf[i]; o = rng.randint(0, len(ids) - W + 1) if len(ids) > W else 0; segs.append(ids[o:o + W])
        ids, m = batch(segs)
        with torch.autocast("cuda", dtype=torch.float16): logits = model(input_ids=ids, attention_mask=m).logits
        loss = Fn.cross_entropy(logits.float(), yf[idx].cuda(), weight=w); assert torch.isfinite(loss), "non-finite loss"
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        if (s // BS) % (10 if SMOKE else 500) == 0: print(f"ep {ep} {s}/{len(perm)} loss {loss.item():.4f}", flush=True)
    f1 = f1_score(y(hold).numpy(), predict(Xh, 3)[0].argmax(1), average="macro"); print("epoch", ep, "holdout macro-F1 (<=3 windows)", round(f1, 4), flush=True)
    if f1 > best: best, state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(state); del opt, sch; torch.cuda.empty_cache()
for nm, X in (("val", Xv), ("test", Xt)):
    mean, ext = predict(X); np.save(f"encwin_{nm}.npy", mean); np.save(f"encwin_x_{nm}.npy", ext)
    if nm == "val": print("VAL macro-F1 (all windows, mean prob)", round(f1_score(y(val).numpy(), mean.argmax(1), average="macro"), 4), flush=True)
print("saved")
