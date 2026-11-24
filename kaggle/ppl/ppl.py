"""GPU: per-segment (intro/body/concl) LM statistics from Qwen2.5-0.5B -> ppl_feats.parquet (Binoculars/TempTest-style signals)."""
import glob, re, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
SMOKE = False  # 300 texts; set False for the full run
find = lambda n: glob.glob(f"/kaggle/input/**/{n}", recursive=True)[0]
tr, te, sp = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv")), pd.read_csv(find("split.csv"))
assert (tr.id.values == sp.id.values).all()
tr["use"] = sp.use.values; te["use"] = "test"
SENT = re.compile(r"(?<=[.!?])\s+")
def segment(text, frac=0.2, min_sent=5):
    s = [x for x in SENT.split(text) if x.strip()]
    if len(s) < min_sent: return None
    k = max(1, round(frac * len(s))); return " ".join(s[:k]), " ".join(s[k:-k]), " ".join(s[-k:])

d = pd.concat([tr[tr.use.isin(["fit", "val"])], te])[["id", "text"]].reset_index(drop=True)
if SMOKE: d = d.sample(300, random_state=0).reset_index(drop=True)
W, MB = 256, 8  # window length in tokens, micro-batch
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16).cuda().eval()

# one window per segment: intro = first W tokens, conclusion = last W, body = centred W (head-only would drop the ending)
wins = []  # (row, seg, ids)
for r, t in enumerate(d.text):
    seg = segment(t)
    if seg is None: continue
    for k, s in enumerate(seg):
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) < 8: continue
        ids = ids[:W] if k == 0 else ids[-W:] if k == 2 else ids[max(0, (len(ids) - W) // 2):][:W]
        wins.append((r, k, ids))
wins.sort(key=lambda w: -len(w[2]))
STAT = ["lp_mean", "lp_std", "ent", "logrank", "ratio"]
out = np.full((len(d), 3, len(STAT)), np.nan, np.float32)
with torch.no_grad():
    for i in range(0, len(wins), MB):
        b = wins[i:i + MB]; L = max(len(w[2]) for w in b)
        ids = torch.full((len(b), L), tok.pad_token_id); m = torch.zeros((len(b), L), dtype=torch.long)
        for j, w in enumerate(b): ids[j, :len(w[2])] = torch.tensor(w[2]); m[j, :len(w[2])] = 1
        ids, m = ids.cuda(), m.cuda()
        lp = torch.log_softmax(model(input_ids=ids, attention_mask=m).logits[:, :-1].float(), -1)
        tlp = lp.gather(-1, ids[:, 1:, None]).squeeze(-1)
        ent = -(lp.exp() * lp).sum(-1); lr = torch.log1p((lp > tlp[..., None]).sum(-1).float())
        mk = m[:, 1:].float(); n = mk.sum(1)
        mean = lambda x: (x * mk).sum(1) / n
        mu = mean(tlp); sd = (((tlp - mu[:, None]) ** 2 * mk).sum(1) / n).sqrt()
        res = torch.stack([mu, sd, mean(ent), mean(lr), -mu / mean(ent)], 1).cpu().numpy()
        for j, w in enumerate(b): out[w[0], w[1]] = res[j]
        if (i // MB) % 500 == 0: print(i, len(wins), flush=True)

cols = {f"{s}_ppl_{n}": out[:, k, c] for k, s in enumerate(["intro", "body", "concl"]) for c, n in enumerate(STAT)}
F = pd.DataFrame(cols)
for n in STAT: F[f"x_ppl_{n}_std"] = F[[f"{s}_ppl_{n}" for s in ("intro", "body", "concl")]].std(axis=1)
F["x_ppl_lp_mean_cd"] = F.concl_ppl_lp_mean - F.intro_ppl_lp_mean
F.insert(0, "id", d.id.values)
assert not np.isinf(F.drop(columns="id").values).any(), "inf in features"
print(F.shape, "NaN share", F.isna().mean().mean())
assert F.isna().mean().mean() < 0.15, "too many NaN (fp16 problems or segmentation)"
F.to_parquet("ppl_feats.parquet", index=False)
