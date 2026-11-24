"""Decoder-LLM text classifier: Qwen3.5-0.8B (fallback Qwen2.5-0.5B; open weights) + sequence-classification head on the last token, head 128 + tail 382 tokens.
Usage: python src/llm_cls.py <n_extra> <seed> <tag> [--smoke]   -> preds/qwen<tag>_{val,test}.npy (order human/ai/fusion), models/qwen<tag>.pt"""
import sys, os, numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
sys.path.insert(0, "src"); from common import load, LABELS

CANDIDATES = ["Qwen/Qwen2.5-0.5B"]  # Qwen3.5-0.8B OOMed at SMOKE on the 16 GB card (30 GB allocated: hybrid linear-attention fallback path), so the plain Qwen2.5 is used
N_EXTRA, SEED, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; SMOKE = "--smoke" in sys.argv
HEAD, TAIL, EPOCHS, BS, LR, INF_BS = 128, 382, 2, 8, 1e-5, 16
fit, val, te = load()
if N_EXTRA:  # extra training rows from the unused train fold, never val
    tr = pd.read_csv("data/raw/train.csv"); use = pd.read_csv("data/split.csv").use.values
    fit = pd.concat([fit, tr[use == "unused"].sample(N_EXTRA, random_state=SEED)])
if SMOKE: fit, val, te = fit.sample(300, random_state=0), val.sample(100, random_state=0), te.head(100)
hold = fit.sample(frac=0.1, random_state=0); fit = fit.drop(hold.index)  # epoch selection on a holdout of fit, never val
print("train", len(fit), "holdout", len(hold), "val", len(val), "test", len(te), flush=True)

for NAME in CANDIDATES:
    try:
        tok = AutoTokenizer.from_pretrained(NAME); model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=3, dtype=torch.float32); break
    except Exception as e: print("cannot load", NAME, repr(e)[:300], flush=True)
else: raise SystemExit("no candidate model could be loaded")
print("using", NAME, "params", sum(p.numel() for p in model.parameters()) / 1e6, "M", flush=True)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
def enc(df, step=4000):  # no special tokens: the classification head reads the last real token (right padding)
    return [np.asarray(x[:HEAD] + x[-TAIL:] if len(x) > HEAD + TAIL else x, np.int64) for i in range(0, len(df), step) for x in tok(list(df.text.iloc[i:i + step]), add_special_tokens=False).input_ids]
y = lambda df: torch.tensor(df.label.map(LABELS.index).values)
Xf, Xh, Xv, Xt = enc(fit), enc(hold), enc(val), enc(te); print("tokenized", flush=True)
def batch(X, idx):
    L = max(len(X[i]) for i in idx); ids = torch.full((len(idx), L), PAD); m = torch.zeros_like(ids)
    for j, i in enumerate(idx): ids[j, :len(X[i])] = torch.from_numpy(X[i]); m[j, :len(X[i])] = 1
    return ids.cuda(), m.cuda()

model = model.cuda()
for c in (model.config, getattr(model.config, "text_config", model.config)): c.pad_token_id = PAD; c.use_cache = False
model.gradient_checkpointing_enable(); model.get_input_embeddings().weight.requires_grad_(False)  # 136M embedding params: no grads / Adam state
yf = y(fit); w = torch.tensor(len(yf) / (3 * np.bincount(yf.numpy(), minlength=3)), dtype=torch.float).cuda()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
steps = EPOCHS * (len(Xf) // BS); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
def predict(X):
    model.eval(); order = np.argsort([len(x) for x in X]); P = np.zeros((len(X), 3), np.float32)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(X), INF_BS):
            idx = order[i:i + INF_BS]; ids, m = batch(X, idx)
            P[idx] = torch.softmax(model(input_ids=ids, attention_mask=m).logits.float(), -1).cpu().numpy()
            if (i // INF_BS) % 400 == 0: print(f"predict {i}/{len(X)}", flush=True)
    model.train(); torch.cuda.empty_cache(); return P

best, state = -1, None
for ep in range(EPOCHS):
    perm = np.random.RandomState(ep + 100 * SEED).permutation(len(Xf))
    for s in range(0, len(perm) - BS + 1, BS):
        idx = perm[s:s + BS]; ids, m = batch(Xf, idx)
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(input_ids=ids, attention_mask=m).logits
        loss = Fn.cross_entropy(logits.float(), yf[idx].cuda(), weight=w); assert torch.isfinite(loss), "non-finite loss"
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if (s // BS) % (10 if SMOKE else 500) == 0: print(f"ep {ep} {s}/{len(perm)} loss {loss.item():.4f}", flush=True)
    f1 = f1_score(y(hold).numpy(), predict(Xh).argmax(1), average="macro"); print("epoch", ep, "holdout macro-F1", round(f1, 4), flush=True)
    if f1 > best: best, state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
os.makedirs("models", exist_ok=True); torch.save(state, f"models/qwen{TAG}.pt")  # weights on disk BEFORE inference
model.load_state_dict(state); del opt, sch; torch.cuda.empty_cache()
Pv = predict(Xv); print("VAL macro-F1", round(f1_score(y(val).numpy(), Pv.argmax(1), average="macro"), 4), flush=True)
np.save(f"preds/qwen{TAG}_val.npy", Pv); np.save(f"preds/qwen{TAG}_test.npy", predict(Xt)); print("saved", TAG)
