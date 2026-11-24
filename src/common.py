"""Shared loading + sentence-level segmentation (intro / body / conclusion). Reused by every script."""
import re, pandas as pd

LABELS = ["human", "ai", "fusion"]
N_FIT = 50_000
SENT = re.compile(r"(?<=[.!?])\s+")  # texts contain no newlines, so segment by sentences


def load():
    """-> fit, val, test DataFrames. `use` column of data/split.csv is written once (fit = baseline's 50k sample)."""
    tr = pd.read_csv("data/raw/train.csv")
    split = pd.read_csv("data/split.csv")
    assert (split.id.values == tr.id.values).all()
    if "use" not in split:
        use = pd.Series("unused", index=tr.index)
        use[tr[split.fold.values == "val"].index] = "val"
        use[tr[split.fold.values == "train"].sample(N_FIT, random_state=42).index] = "fit"  # same call as baseline.py
        split["use"] = use.values
        split.to_csv("data/split.csv", index=False)
    use = split.use.values
    return tr[use == "fit"], tr[use == "val"], pd.read_csv("data/raw/test.csv")


def segment(text, frac=0.2, min_sent=5):
    """-> (intro, body, conclusion) strings: first/last `frac` of sentences (>=1 each); None if fewer than `min_sent`."""
    s = [x for x in SENT.split(text) if x.strip()]
    if len(s) < min_sent:
        return None
    k = max(1, round(frac * len(s)))
    return " ".join(s[:k]), " ".join(s[k:-k]), " ".join(s[-k:])


if __name__ == "__main__":
    t10 = " ".join(f"Sentence {i}." for i in range(10))
    i, b, c = segment(t10)
    assert (i.count("."), b.count("."), c.count(".")) == (2, 6, 2) and segment("A. B. C.") is None
    fit, val, te = load()
    base = pd.read_csv("data/raw/train.csv"); sp = pd.read_csv("data/split.csv")
    assert set(fit.id) == set(base[sp.fold.values == "train"].sample(N_FIT, random_state=42).id)
    assert not set(fit.id) & set(val.id)
    print("ok", len(fit), len(val), len(te))
