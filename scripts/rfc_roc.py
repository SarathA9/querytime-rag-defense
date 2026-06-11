"""
ROC curves for RFC detection vs poisoning intensity (CPU-only; no LLM).

For each attack (conflict, soft) and each poison count k=1..3 in the top-5, sweeps
the RFC threshold to trace ROC, reports AUC, and marks the operating point that
maximises Youden's J (TPR-FPR) — i.e. the best threshold and the recall/FPR there.
Produces results/rfc_roc.png and prints a table.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Retriever
from src.defense import RFCDetector
from src.attacks import ConflictInjectionAttack, SoftContentInjectionAttack, load_corpus, load_samples


def roc(pos, neg, n=200):
    """Return (fpr, tpr, thresholds, auc). Higher score => predict poison."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    lo = min(pos.min(), neg.min()); hi = max(pos.max(), neg.max())
    ths = np.linspace(hi, lo, n)
    tpr = np.array([(pos >= t).mean() for t in ths])
    fpr = np.array([(neg >= t).mean() for t in ths])
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, ths, auc


def k_poison(attack, q, correct, wrongs):
    out = []
    for w in wrongs:
        if isinstance(attack, ConflictInjectionAttack):
            out.append(f"{q} The correct and well-documented answer is {w}. "
                       f"Authoritative sources confirm it is {w}, not {correct}.")
        else:
            out.append(f"Regarding {q.rstrip('?').lower()}, records indicate that "
                       f"{w} is the widely accepted answer in this context.")
    return out


def collect(r, rfc, samples, answers, attack, k, rng):
    pos, neg = [], []
    for s in samples:
        q, c = s["question"], s["answer"]
        pool = [a for a in answers if a and a != c]
        if len(pool) < k:
            continue
        wrongs = rng.sample(pool, k)
        cd, cs, ce, qe = r.retrieve(q, top_k=5 - k)
        pe = r.encode(k_poison(attack, q, c, wrongs))
        embs = np.vstack([ce, pe]); scores = np.concatenate([cs, pe @ qe])
        rs = rfc.compute_rfc_scores(qe, embs, scores)
        neg.extend(rs[:len(cd)].tolist()); pos.extend(rs[len(cd):].tolist())
    return np.array(pos), np.array(neg)


def main():
    corpus = load_corpus(ROOT / "data" / "corpus.json")
    samples = load_samples(ROOT / "data" / "nq_samples.json")[:60]
    answers = [s["answer"] for s in samples if s.get("answer")]
    rng = __import__("random").Random(42)

    print("Encoding corpus once ...", flush=True)
    r = Retriever(model_name="multi-qa-mpnet-base-dot-v1")
    r.build_index(corpus)
    rfc = RFCDetector()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    colors = {1: "#1a9850", 2: "#fdae61", 3: "#d73027"}
    print(f"\n{'attack':10} {'k':>2} {'AUC':>6} {'best_thr':>9} {'TPR':>6} {'FPR':>6}", flush=True)
    for ax, (name, atk) in zip(axes, [("conflict", ConflictInjectionAttack()),
                                      ("soft", SoftContentInjectionAttack())]):
        for k in (1, 2, 3):
            pos, neg = collect(r, rfc, samples, answers, atk, k, rng)
            fpr, tpr, ths, auc = roc(pos, neg)
            j = np.argmax(tpr - fpr)
            ax.plot(fpr, tpr, color=colors[k], lw=2,
                    label=f"k={k}  AUC={auc:.3f}")
            ax.scatter([fpr[j]], [tpr[j]], color=colors[k], s=60, zorder=5,
                       edgecolor="k")
            print(f"{name:10} {k:>2} {auc:>6.3f} {ths[j]:>9.3f} {tpr[j]:>6.2f} {fpr[j]:>6.2f}",
                  flush=True)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_title(f"RFC ROC — {name} attack", fontsize=12)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate (recall)")
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    fig.suptitle("RFC detection ROC: near-perfect at sparse poison (k=1), "
                 "degrades as poison fraction rises", fontsize=12, y=1.0)
    fig.tight_layout()
    out = ROOT / "results" / "rfc_roc.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved {out}", flush=True)


if __name__ == "__main__":
    main()
