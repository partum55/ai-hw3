"""GPU: fine-tune a multilingual encoder, head+tail 512 tokens -> enc_val.npy / enc_test.npy (probabilities, order human/ai/fusion)."""
import glob, numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
SMOKE = False  # 300 texts; set False for the full run
find = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
tr, te, sp = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv")), pd.read_csv(find("split.csv"))
assert (tr.id.values == sp.id.values).all()
tr["use"] = sp.use.values
LABELS = ["human", "ai", "fusion"]
N_EXTRA, TAG, SEED = 0, "", 1  # extra training rows from the unused train fold (never val); TAG suffixes the output names
BACKBONES = ("microsoft/mdeberta-v3-base", "xlm-roberta-base")  # tried in order; next one is the fallback on non-finite loss
fit, val = tr[tr.use == "fit"], tr[tr.use == "val"]
if N_EXTRA: fit = pd.concat([fit, tr[tr.use == "unused"].sample(N_EXTRA, random_state=SEED)])
if SMOKE: fit, val, te = fit.sample(300, random_state=0), val.sample(100, random_state=0), te.head(100)
hold = fit.sample(frac=0.1, random_state=0); fit = fit.drop(hold.index)  # epoch selection on a holdout of fit, never val
HEAD, TAIL, EPOCHS, BS, LR = 128, 382, 2, 8, 2e-5

def run(name):
    tok = AutoTokenizer.from_pretrained(name)
    def enc(df):  # head+tail truncation keeps the intro AND the conclusion of long texts
        ids = tok(list(df.text), add_special_tokens=False).input_ids
        return [[tok.cls_token_id] + (x[:HEAD] + x[-TAIL:] if len(x) > HEAD + TAIL else x) + [tok.sep_token_id] for x in ids]
    y = lambda df: torch.tensor(df.label.map(LABELS.index).values)
    Xf, Xh, Xv, Xt = enc(fit), enc(hold), enc(val), enc(te)
    def batch(X, idx):
        L = max(len(X[i]) for i in idx); ids = torch.full((len(idx), L), tok.pad_token_id); m = torch.zeros_like(ids)
        for j, i in enumerate(idx): ids[j, :len(X[i])] = torch.tensor(X[i]); m[j, :len(X[i])] = 1
        return ids.cuda(), m.cuda()
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3, torch_dtype=torch.float32).cuda()
    yf = y(fit); w = torch.tensor(len(yf) / (3 * np.bincount(yf.numpy(), minlength=3)), dtype=torch.float).cuda()  # class weights
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * (len(Xf) // BS); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps); scaler = torch.amp.GradScaler()
    def predict(X):
        model.eval(); order = np.argsort([len(x) for x in X]); P = np.zeros((len(X), 3), np.float32)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(X), 32):
                idx = order[i:i + 32]; P[idx] = torch.softmax(model(**dict(zip(("input_ids", "attention_mask"), batch(X, idx)))).logits.float(), -1).cpu().numpy()
        model.train(); torch.cuda.empty_cache(); return P
    best, state = -1, None
    for ep in range(EPOCHS):
        perm = np.random.RandomState(ep).permutation(len(Xf))
        for s in range(0, len(perm) - BS + 1, BS):
            idx = perm[s:s + BS]; ids, m = batch(Xf, idx)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=m).logits
            loss = Fn.cross_entropy(logits.float(), yf[idx].cuda(), weight=w)
            if not torch.isfinite(loss): return None  # fp16 overflow -> caller falls back to another backbone
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            if (s // BS) % 200 == 0: print(name, ep, s, len(perm), float(loss), flush=True)
        f1 = f1_score(y(hold).numpy(), predict(Xh).argmax(1), average="macro"); print(name, "epoch", ep, "holdout macro-F1", f1, flush=True)
        if f1 > best: best, state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    Pv = predict(Xv); print(name, "VAL macro-F1", f1_score(y(val).numpy(), Pv.argmax(1), average="macro"), flush=True)
    return Pv, predict(Xt)

for name in BACKBONES:
    r = run(name)
    if r is not None: break
    print(name, "hit non-finite loss, falling back", flush=True)
assert r is not None
np.save(f"enc{TAG}_val.npy", r[0]); np.save(f"enc{TAG}_test.npy", r[1]); print("saved", r[0].shape, r[1].shape, "backbone", name)
