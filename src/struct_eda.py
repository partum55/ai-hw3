"""Dash rates per lang/domain and intro/body/conclusion features per class (train fold only, 40k sample)."""
import re, numpy as np, pandas as pd

tr = pd.read_csv("data/raw/train.csv"); fold = pd.read_csv("data/split.csv").fold.values
d = tr[fold == "train"].sample(40_000, random_state=0).reset_index(drop=True)

per1k = lambda s, p: s.str.count(p) / s.str.len() * 1000
d["em"], d["en"] = per1k(d.text, "—"), per1k(d.text, "–")
print("em-dash '—' per 1k chars, mean by lang x label\n", d.groupby(["lang", "label"]).em.mean().unstack().round(3))
print("\nen-dash '–' per 1k chars, mean by lang x label\n", d.groupby(["lang", "label"]).en.mean().unstack().round(3))
print("\nem-dash per 1k chars by domain x label\n", d.groupby(["domain", "label"]).em.mean().unstack().round(3))

SENT = re.compile(r"(?<=[.!?])\s+")
END = re.compile(r"in conclusion|ultimately|overall|in summary|to sum up|reminds us|serves as|zusammenfassend|abschließend|letztlich|insgesamt|fazit", re.I)
def feats(t):
    s = [x for x in SENT.split(t) if x.strip()]
    if len(s) < 6: return [np.nan] * 8
    th = np.array_split(np.arange(len(s)), 3)
    ml = [np.mean([len(s[i].split()) for i in ix]) for ix in th]            # mean sentence length per third
    cm = [t_.count(",") / max(len(t_), 1) * 1000 for t_ in (" ".join(s[i] for i in ix) for ix in th)]  # commas per 1k chars
    return [*ml, *cm[:1], cm[2], np.std(ml), float(bool(END.search(" ".join(s[-2:])))), float("?" in s[0])]
cols = ["len_intro", "len_body", "len_concl", "comma_intro", "comma_concl", "len_std_across", "concl_marker", "intro_question"]
F = pd.DataFrame([feats(t) for t in d.text], columns=cols); F["label"] = d.label; F["lang"] = d.lang
print("\nsegment features, mean by lang x label (rows with >=6 sentences)")
print(F.groupby(["lang", "label"])[cols].mean().round(3).to_string())
