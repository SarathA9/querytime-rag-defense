"""
PCA embedding analysis: mechanistic explanation of when/why RFC works.

Produces results/pca_rfc_analysis.png with three panels:
  (A) k=1 regime  - query + clean docs + 1 poison in 2-D PCA. The poison is near
      the query (retrieved) but off the clean centroid -> large RFC -> detected.
  (B) k=3 regime  - poison dominates the retrieved set, so the "centroid of the
      others" is itself poison; poison looks faithful -> small RFC -> missed.
  (C) RFC score distributions (poison vs clean) at k=1 vs k=3, aggregated over
      many queries - shows the separation collapsing as poison fraction rises.

CPU-only; no LLM. Mirrors the soft-content attack (the stealthy case).
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Retriever
from src.defense import RFCDetector
from src.attacks import load_corpus, load_samples

THR = 0.1


def soft_poison(q, wrong, correct):
    return (f"Regarding {q.rstrip('?').lower()}, records indicate that {wrong} is "
            f"the widely accepted answer in this context.")


def panel_scatter(ax, qemb, clean_embs, pois_embs, title):
    """2-D PCA of one query's retrieved context; annotate RFC geometry."""
    pts = np.vstack([qemb[None], clean_embs, pois_embs])
    p2 = PCA(n_components=2, random_state=0).fit_transform(pts)
    q2, c2, x2 = p2[0], p2[1:1 + len(clean_embs)], p2[1 + len(clean_embs):]
    ax.scatter(*c2.T, c="#2c7fb8", s=90, label="clean docs", edgecolor="k", zorder=3)
    ax.scatter(*x2.T, c="#d7301f", s=110, marker="^", label="poison docs",
               edgecolor="k", zorder=3)
    ax.scatter(*q2, c="#fec44f", s=260, marker="*", label="query", edgecolor="k", zorder=4)
    # centroid of "others" for the first poison doc (what RFC compares it to)
    others = np.vstack([clean_embs, pois_embs[1:]]) if len(pois_embs) > 1 else clean_embs
    o2 = PCA(n_components=2, random_state=0).fit(pts).transform(others.mean(0)[None])[0]
    ax.scatter(*o2, c="#54278f", s=140, marker="X", label="centroid of others", zorder=4)
    ax.plot([x2[0,0], o2[0]], [x2[0,1], o2[1]], "--", c="#54278f", lw=1.5, zorder=2)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=7, loc="best")


def main():
    corpus = load_corpus(ROOT / "data" / "corpus.json")
    samples = load_samples(ROOT / "data" / "nq_samples.json")[:60]
    answers = [s["answer"] for s in samples if s.get("answer")]
    rng = __import__("random").Random(42)

    print("Encoding corpus once ...", flush=True)
    r = Retriever(model_name="multi-qa-mpnet-base-dot-v1")
    r.build_index(corpus)
    rfc = RFCDetector(threshold=THR)

    # --- pick an illustrative query for panels A/B ---
    s = samples[0]; q, correct = s["question"], s["answer"]
    wrongs = rng.sample([a for a in answers if a != correct], 3)

    # k=1
    cd1, cs1, ce1, qe = r.retrieve(q, top_k=4)
    pe1 = r.encode([soft_poison(q, wrongs[0], correct)])
    # k=3
    cd3, cs3, ce3, _ = r.retrieve(q, top_k=2)
    pe3 = r.encode([soft_poison(q, w, correct) for w in wrongs])

    # --- panel C: aggregate RFC scores at k=1 and k=3 ---
    agg = {1: {"p": [], "c": []}, 3: {"p": [], "c": []}}
    for samp in samples:
        qq, cc = samp["question"], samp["answer"]
        ws = rng.sample([a for a in answers if a != cc], 3)
        for k in (1, 3):
            cd, cs, ce, qe2 = r.retrieve(qq, top_k=5 - k)
            pd = [soft_poison(qq, ws[i], cc) for i in range(k)]
            pe = r.encode(pd)
            embs = np.vstack([ce, pe]); scores = np.concatenate([cs, pe @ qe2])
            rs = rfc.compute_rfc_scores(qe2, embs, scores)
            agg[k]["c"].extend(rs[:len(cd)].tolist())
            agg[k]["p"].extend(rs[len(cd):].tolist())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    panel_scatter(axes[0], qe, ce1, pe1,
                  "(A) k=1: poison off the clean centroid → high RFC → DETECTED")
    panel_scatter(axes[1], qe, ce3, pe3,
                  "(B) k=3: poison dominates → centroid is poison → low RFC → MISSED")
    ax = axes[2]
    for k, col in [(1, "#2c7fb8"), (3, "#d7301f")]:
        ax.hist(agg[k]["p"], bins=20, alpha=0.5, color=col, label=f"poison k={k}")
        ax.hist(agg[k]["c"], bins=20, alpha=0.25, color=col, histtype="step",
                lw=2, label=f"clean k={k}")
    ax.axvline(THR, c="k", ls="--", lw=1, label=f"threshold={THR}")
    ax.set_title("(C) RFC score separation collapses as poison rises", fontsize=11)
    ax.set_xlabel("RFC score (retrieval − faithfulness)"); ax.set_ylabel("count")
    ax.legend(fontsize=7)
    fig.suptitle("PCA / RFC mechanism: RFC detects sparse poison, fails when poison dominates",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = ROOT / "results" / "pca_rfc_analysis.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"✓ Saved {out}", flush=True)
    for k in (1, 3):
        p, c = np.array(agg[k]["p"]), np.array(agg[k]["c"])
        print(f"  k={k}: poisonRFC mean={p.mean():+.3f}  cleanRFC mean={c.mean():+.3f}  "
              f"gap={p.mean()-c.mean():+.3f}", flush=True)


if __name__ == "__main__":
    main()
