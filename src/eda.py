"""EDA: class balance by lang/domain, text lengths, one example per class. Figures -> images/."""
import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

tr = pd.read_csv("data/raw/train.csv")
tr["len"] = tr.text.str.len()
tr["words"] = tr.text.str.split().str.len()
for c in ["lang", "domain"]:
    t = pd.crosstab(tr[c], tr.label, normalize="index").round(3)
    t["n"] = tr[c].value_counts()
    print(f"\nclass share by {c}\n{t}")
print("\nlength (chars) by label\n", tr.groupby("label")["len"].describe().round(0))
print("\nwords by label x lang (median)\n", tr.groupby(["lang", "label"])["words"].median().unstack())

pd.crosstab(tr.domain, tr.label, normalize="index").plot.barh(stacked=True, figsize=(7, 4))
plt.tight_layout(); plt.savefig("images/class_by_domain.png", dpi=120); plt.close()
tr.boxplot(column="words", by="label", figsize=(5, 4)); plt.yscale("log")
plt.tight_layout(); plt.savefig("images/words_by_label.png", dpi=120); plt.close()

print("\nexamples (first 250 chars)")
for lab, g in tr.groupby("label"):
    r = g.sample(1, random_state=0).iloc[0]
    print(f"[{lab}|{r.lang}|{r.domain}] {r.text[:250]!r}")
