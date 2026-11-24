"""Local-GPU version of kaggle/enc/enc.py (head+tail 512 tokens, bf16). Usage: python src/enc_local.py <mdeberta|xlmr> <n_extra> <seed> <tag> [--smoke]
Writes preds/enc<tag>_{val,test}.npy (probabilities, order human/ai/fusion)."""
import sys, numpy as np, pandas as pd, torch, torch.nn.functional as Fn
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
sys.path.insert(0, "src"); from common import load, LABELS

NAMES = {"mdeberta": "microsoft/mdeberta-v3-base", "xlmr": "xlm-roberta-base", "xlmr-large": "xlm-roberta-large", "deberta-v3": "microsoft/deberta-v3-base"}  # deberta-v3 is English-only: use with --en-only
name, N_EXTRA, SEED, TAG = NAMES[sys.argv[1]], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]; SMOKE = "--smoke" in sys.argv; EN = "--en-only" in sys.argv
HEAD, TAIL, EPOCHS, BS, LR = 128, 382, 2, 8, 2e-5
INF_BS = 16  # 32+ x 512 tokens overflowed 16 GB for DeBERTa-style attention (spill to system RAM)
LARGE = "large" in name
if LARGE: BS, LR = 4, 1e-5  # large model: smaller batch (16 GB GPU) and lower lr (large XLM-R diverges easily at 2e-5)

fit, val, te = load()
if EN: fit = fit[fit.lang == "en-EN"]  # English-only training rows (val/test are still predicted in full; the stacker knows `lang`)
if N_EXTRA:  # extra training rows from the unused train fold, never val
    tr = pd.read_csv("data/raw/train.csv"); use = pd.read_csv("data/split.csv").use.values; pool = use == "unused"
    if EN: pool &= (tr.lang == "en-EN").values
    fit = pd.concat([fit, tr[pool].sample(N_EXTRA, random_state=SEED)])
if SMOKE: fit, val, te = fit.sample(300, random_state=0), val.sample(100, random_state=0), te.head(100)
hold = fit.sample(frac=0.1, random_state=0); fit = fit.drop(hold.index)  # epoch selection on a holdout of fit, never val
print(name, "train", len(fit), "holdout", len(hold), "val", len(val), "test", len(te), flush=True)

tok = AutoTokenizer.from_pretrained(name)
def enc(df, step=4000):  # chunked: full-length token lists of 150k texts would not fit in 15 GB RAM
    out = []
    for i in range(0, len(df), step):
        for x in tok(list(df.text[i:i + step]), add_special_tokens=False).input_ids:
            out.append([tok.cls_token_id] + (x[:HEAD] + x[-TAIL:] if len(x) > HEAD + TAIL else x) + [tok.sep_token_id])
    return out
y = lambda df: torch.tensor(df.label.map(LABELS.index).values)
Xf, Xh, Xv, Xt = enc(fit), enc(hold), enc(val), enc(te); print("tokenized", flush=True)
def batch(X, idx):
    L = max(len(X[i]) for i in idx); ids = torch.full((len(idx), L), tok.pad_token_id); m = torch.zeros_like(ids)
    for j, i in enumerate(idx): ids[j, :len(X[i])] = torch.tensor(X[i]); m[j, :len(X[i])] = 1
    return ids.cuda(), m.cuda()

model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=3, dtype=torch.float32).cuda()
if LARGE:  # fit 16 GB: checkpointing + frozen 250k-token embedding matrix (256M of 560M params -> no grads/Adam state for it)
    model.gradient_checkpointing_enable(); model.base_model.embeddings.word_embeddings.weight.requires_grad_(False)
yf = y(fit); w = torch.tensor(len(yf) / (3 * np.bincount(yf.numpy(), minlength=3)), dtype=torch.float).cuda()  # class weights
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
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=m).logits
        loss = Fn.cross_entropy(logits.float(), yf[idx].cuda(), weight=w)
        assert torch.isfinite(loss), "non-finite loss"
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if (s // BS) % (10 if SMOKE else 500) == 0: print(f"ep {ep} {s}/{len(perm)} loss {float(loss):.4f}", flush=True)
    f1 = f1_score(y(hold).numpy(), predict(Xh).argmax(1), average="macro"); print("epoch", ep, "holdout macro-F1", round(f1, 4), flush=True)
    if f1 > best: best, state = f1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
import os; os.makedirs("models", exist_ok=True); torch.save(state, f"models/enc{TAG}.pt")  # weights on disk BEFORE inference
model.load_state_dict(state)
del opt, sch; torch.cuda.empty_cache()  # free AdamW state before inference (GPU memory was ~full)
Pv = predict(Xv); print("VAL macro-F1", round(f1_score(y(val).numpy(), Pv.argmax(1), average="macro"), 4), flush=True)
np.save(f"preds/enc{TAG}_val.npy", Pv); np.save(f"preds/enc{TAG}_test.npy", predict(Xt)); print("saved", TAG)
