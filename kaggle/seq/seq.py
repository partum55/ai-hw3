"""GPU: sentence embeddings (first 32 + last 32 sentences) -> BiGRU + attention pooling -> seq_val.npy / seq_test.npy."""
import glob, re, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
SMOKE = True  # 300 texts; set False for the full run
find = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
tr, te, sp = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv")), pd.read_csv(find("split.csv"))
assert (tr.id.values == sp.id.values).all()
tr["use"] = sp.use.values
LABELS = ["human", "ai", "fusion"]; K = 32
fit, val = tr[tr.use == "fit"], tr[tr.use == "val"]
if SMOKE: fit, val, te = fit.sample(300, random_state=0), val.sample(100, random_state=0), te.head(100)
SENT = re.compile(r"(?<=[.!?])\s+")
st = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda"); st.max_seq_length = 64; st.half()

def embed(df):
    """-> X [N, 2K, 386] fp16 (384 emb + relative position + valid flag), mask [N, 2K]"""
    sents, own = [], []
    for r, t in enumerate(df.text):
        s = [x for x in SENT.split(t) if x.strip()] or [t]; n = len(s)
        keep = list(range(n)) if n <= 2 * K else list(range(K)) + list(range(n - K, n))
        for i in keep: sents.append(s[i]); own.append((r, i / max(n - 1, 1)))
    E = st.encode(sents, batch_size=256, convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    X = np.zeros((len(df), 2 * K, 386), np.float16); cnt = np.zeros(len(df), int)
    for e, (r, pos) in zip(E, own):
        X[r, cnt[r], :384] = e; X[r, cnt[r], 384] = pos; X[r, cnt[r], 385] = 1; cnt[r] += 1
    return torch.tensor(X), torch.tensor(X[:, :, 385] > 0)

class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.gru = nn.GRU(386, 128, batch_first=True, bidirectional=True); s.att = nn.Linear(256, 1); s.out = nn.Sequential(nn.Dropout(0.3), nn.Linear(256, 3))
    def forward(s, x, m):
        h, _ = s.gru(x); a = s.att(h).squeeze(-1).masked_fill(~m, -1e4).softmax(1)
        return s.out((a[..., None] * h).sum(1))

hold = fit.sample(frac=0.1, random_state=0); fit = fit.drop(hold.index)  # early stopping on a holdout of fit, never val
(Xf, Mf), (Xh, Mh), (Xv, Mv), (Xt, Mt) = map(embed, (fit, hold, val, te))
y = lambda df: torch.tensor(df.label.map(LABELS.index).values)
yf, yh, yv = y(fit), y(hold), y(val)
net = Net().cuda(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
w = torch.tensor(len(yf) / (3 * np.bincount(yf.numpy(), minlength=3)), dtype=torch.float).cuda()
def predict(X, M):
    net.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(X), 512): out.append(net(X[i:i + 512].cuda().float(), M[i:i + 512].cuda()).softmax(-1).cpu())
    net.train(); return torch.cat(out).numpy()
best, state, bad = -1, None, 0
for ep in range(30):
    perm = torch.randperm(len(Xf))
    for i in range(0, len(perm), 128):
        idx = perm[i:i + 128]; loss = nn.functional.cross_entropy(net(Xf[idx].cuda().float(), Mf[idx].cuda()), yf[idx].cuda(), weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    f1 = f1_score(yh.numpy(), predict(Xh, Mh).argmax(1), average="macro"); print("epoch", ep, "holdout macro-F1", round(f1, 4), flush=True)
    if f1 > best: best, state, bad = f1, {k: v.clone() for k, v in net.state_dict().items()}, 0
    else:
        bad += 1
        if bad >= 4: break
net.load_state_dict(state); Pv = predict(Xv, Mv); print("VAL macro-F1", f1_score(yv.numpy(), Pv.argmax(1), average="macro"))
np.save("seq_val.npy", Pv); np.save("seq_test.npy", predict(Xt, Mt))
