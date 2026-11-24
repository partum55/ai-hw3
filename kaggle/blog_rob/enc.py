"""GPU (Kaggle): short-text / blog specialist. Trains ONLY on blog rows (fit + unused; never val), head 256 tokens,
then predicts ALL val/test rows (the stacker uses the domain one-hot to decide when to trust it) -> {TAG}_{val,test}.npy"""
import glob, numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
SMOKE = False  # 300 texts; set False for the full run
NAME, TAG, LR = "roberta-large", "blogrob", 1e-5  # roberta-large variant: LR 1e-5
MAXLEN, EPOCHS, BS, BLOG = 256, 3, 16, "blog_authorship_corpus"
find = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
tr, te, sp = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv")), pd.read_csv(find("split.csv"))
assert (tr.id.values == sp.id.values).all()
tr["use"] = sp.use.values
LABELS = ["human", "ai", "fusion"]
trn, val = tr[(tr.domain == BLOG) & tr.use.isin(["fit", "unused"])], tr[tr.use == "val"]
if SMOKE: trn, val, te = trn.sample(300, random_state=0), val.sample(100, random_state=0), te.head(100)
hold = trn.sample(frac=0.1, random_state=0); trn = trn.drop(hold.index)  # epoch selection on a holdout of the TRAIN rows, never val
print(NAME, "train blog rows", len(trn), "holdout", len(hold), "val rows (all domains)", len(val), "test", len(te), flush=True)

tok = AutoTokenizer.from_pretrained(NAME)
enc = lambda df, step=4000: [x for i in range(0, len(df), step) for x in tok(list(df.text.iloc[i:i + step]), truncation=True, max_length=MAXLEN).input_ids]
y = lambda df: torch.tensor(df.label.map(LABELS.index).values)
Xf, Xh, Xv, Xt = enc(trn), enc(hold), enc(val), enc(te)
def batch(X, idx):
    L = max(len(X[i]) for i in idx); ids = torch.full((len(idx), L), tok.pad_token_id); m = torch.zeros_like(ids)
    for j, i in enumerate(idx): ids[j, :len(X[i])] = torch.tensor(X[i]); m[j, :len(X[i])] = 1
    return ids.cuda(), m.cuda()

model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=3, torch_dtype=torch.float32).cuda()
yf = y(trn); w = torch.tensor(len(yf) / (3 * np.bincount(yf.numpy(), minlength=3)), dtype=torch.float).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * (len(Xf) // BS); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps); scaler = torch.amp.GradScaler()
def predict(X):
    model.eval(); order = np.argsort([len(x) for x in X]); P = np.zeros((len(X), 3), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(X), 64):
            idx = order[i:i + 64]; ids, m = batch(X, idx); P[idx] = torch.softmax(model(input_ids=ids, attention_mask=m).logits.float(), -1).cpu().numpy()
    model.train(); torch.cuda.empty_cache(); return P

best, state = -1, None
for ep in range(EPOCHS):
    perm = np.random.RandomState(ep).permutation(len(Xf))
    for s in range(0, len(perm) - BS + 1, BS):
        idx = perm[s:s + BS]; ids, m = batch(Xf, idx)
        with torch.autocast("cuda", dtype=torch.float16): logits = model(input_ids=ids, attention_mask=m).logits
        loss = Fn.cross_entropy(logits.float(), yf[idx].cuda(), weight=w); assert torch.isfinite(loss), "non-finite loss"
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        if (s // BS) % (10 if SMOKE else 300) == 0: print(f"ep {ep} {s}/{len(perm)} loss {loss.item():.4f}", flush=True)
    f1 = f1_score(y(hold).numpy(), predict(Xh).argmax(1), average="macro"); print("epoch", ep, "holdout (blog) macro-F1", round(f1, 4), flush=True)
    if f1 > best: best, state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(state)
Pv = predict(Xv); isb = (val.domain == BLOG).values
print("VAL macro-F1 blog rows", round(f1_score(y(val).numpy()[isb], Pv[isb].argmax(1), average="macro"), 4), "| all rows", round(f1_score(y(val).numpy(), Pv.argmax(1), average="macro"), 4), flush=True)
np.save(f"{TAG}_val.npy", Pv); np.save(f"{TAG}_test.npy", predict(Xt)); print("saved")
