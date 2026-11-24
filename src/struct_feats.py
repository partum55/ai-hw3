"""Per-segment (intro/body/concl) + whole-text CPU features -> data/feats_struct.parquet. Ideas: Tripto et al. EMNLP 2025.
NER is skipped (slowest spaCy component, tiny expected signal) -- noted as deviation in the report."""
import re, sys, numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, "src"); from common import load, segment, SENT

POS = ["ADJ", "ADV", "NOUN", "VERB", "PRON", "DET", "ADP", "AUX", "PROPN"]
PUNCT = {"comma": ",", "semi": ";", "colon": ":", "excl": "!", "ques": "?", "emdash": "—", "endash": "–", "quote": '"', "hyph": " - "}
END = re.compile(r"in conclusion|ultimately|overall|in summary|to sum up|reminds us|serves as|zusammenfassend|abschließend|letztlich|insgesamt|fazit", re.I)
BASE = None  # list of per-segment feature names, filled on first call


def cheap(t):
    """regex-only features: sentence stats + punctuation rates per 1k chars."""
    s = [x for x in SENT.split(t) if x.strip()]; w = np.array([len(x.split()) for x in s] or [0])
    f = {"n_sent": len(s), "sent_mean": w.mean(), "sent_std": w.std(), "n_chars": len(t)}
    f.update({k: t.count(p) / max(len(t), 1) * 1000 for k, p in PUNCT.items()})
    return f


def init():
    global NLP, ts
    import spacy, textstat as ts
    NLP = {l: spacy.load(m, disable=["parser", "ner", "lemmatizer"]) for l, m in (("en", "en_core_web_sm"), ("de", "de_core_news_sm"))}


def rich(doc, text, lang):
    toks = [t for t in doc if not (t.is_space or t.is_punct)]; n = max(len(toks), 1)
    f = {"brunet": len({t.lower_ for t in toks}) / n ** 0.165 if toks else np.nan, "stop": sum(t.is_stop for t in toks) / n}
    f.update({f"pos_{p}": sum(t.pos_ == p for t in toks) / n for p in POS})
    ts.set_lang(lang); f["flesch"] = ts.flesch_reading_ease(text)
    return f


def one(args):
    t, lang = args; lang = lang[:2]
    out = {f"all_{k}": v for k, v in cheap(t).items()}
    seg = segment(t)
    names = None
    if seg is not None:
        docs = list(NLP[lang].pipe(seg, batch_size=3))
        per = [{**cheap(s), **rich(d, s, lang)} for s, d in zip(seg, docs)]
        for nm, f in zip(("intro", "body", "concl"), per): out.update({f"{nm}_{k}": v for k, v in f.items()})
        for k in per[0]:  # cross-segment heterogeneity
            v = np.array([p[k] for p in per], float); out[f"x_{k}_std"] = v.std(); out[f"x_{k}_cd"] = v[2] - v[0]
        s = [x for x in SENT.split(t) if x.strip()]
        out["concl_marker"] = float(bool(END.search(" ".join(s[-2:])))); out["intro_ques"] = float("?" in s[0]); out["intro_excl"] = float("!" in s[0])
    return out


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    fit, val, te = load(); d = pd.concat([fit, val, te])[["id", "text", "lang"]]
    if smoke: d = d.sample(300, random_state=0)
    with Pool(5, initializer=init) as p: rows = p.map(one, zip(d.text, d.lang), chunksize=200)
    F = pd.DataFrame(rows); F.insert(0, "id", d.id.values)
    assert len(F) == len(d) and F.id.is_unique
    print(F.shape, f"NaN share {F.isna().mean().mean():.3f}", f"rows without segments {F.intro_n_sent.isna().mean():.3f}")
    if not smoke: F.to_parquet("data/feats_struct.parquet", index=False)
