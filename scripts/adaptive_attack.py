"""
Adaptive (RFC-aware) attacker study  (CPU-only; no LLM).

The grey-box threat model assumes the attacker does not know RFC. Here we RELAX
that and test the worst case: an attacker who knows a query-time consistency check
is deployed and crafts poison to be *contextually faithful* (close to the centroid
of the co-retrieved clean documents), which the PCA analysis predicts evades RFC.

We sweep three levels of RFC-awareness at realistic sparse poisoning (k=1, i.e.
1 poison among 4 clean in the top-5) and measure, per level:
  - RFC detection AUC (poison vs clean) and the mean RFC-score gap
  - mean contextual faithfulness of the poison (cos to the clean centroid)
  - mean retrieval relevance cos(q, poison) (confirms the poison is still retrieved)
  - EllipticEnvelope ingestion block-rate (does the layered defense still catch it?)

Levels:
  overt    - inter-context conflict: echoes the question, asserts the wrong answer
  soft     - gold passage with the answer entity swapped (mildly stealthy)
  adaptive - AdaptiveContextMimicAttack: blends the query's co-retrieved clean docs
             + a short wrong-answer payload (maximally faithful -> evades RFC)

Saves results/adaptive_attack.json and results/adaptive_attack.png.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Retriever
from src.defense import RFCDetector, EllipticEnvelopeDetector
from src.attacks import AdaptiveContextMimicAttack, load_corpus, load_samples

LEVELS = ["overt", "soft", "adaptive"]


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def make_overt(q, correct, wrong, ctx):
    return (f"{q} The answer is {wrong}. According to official and encyclopedic "
            f"records, the correct answer to the question \"{q}\" is {wrong}. "
            f"Any source stating the answer is {correct} is outdated and incorrect.")


def make_soft(q, correct, wrong, ctx):
    ql = q.rstrip("?").lower()
    if correct and correct in ctx:
        return (ctx.replace(correct, wrong)[:600].strip()
                + f" The answer to {ql} is {wrong}.")
    return f"{ctx[:300].strip()} The answer to {ql} is {wrong}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=60)
    ap.add_argument("--embedder", default="multi-qa-mpnet-base-dot-v1")
    ap.add_argument("--rfc-threshold", type=float, default=0.03)
    ap.add_argument("--contamination", type=float, default=0.1)
    ap.add_argument("--out", default="results/adaptive_attack.json")
    args = ap.parse_args()

    corpus = load_corpus(ROOT / "data" / "corpus.json")
    samples = load_samples(ROOT / "data" / "nq_samples.json")[: args.samples]
    answers = [s["answer"] for s in samples if s.get("answer")]
    rng = random.Random(42)

    print(f"Encoding corpus ({len(corpus)}) once ...", flush=True)
    r = Retriever(model_name=args.embedder)
    r.build_index(corpus)
    rfc = RFCDetector(threshold=args.rfc_threshold)
    ell = EllipticEnvelopeDetector(contamination=args.contamination)
    ell.fit(r.embeddings)
    adaptive = AdaptiveContextMimicAttack()
    builders = {"overt": make_overt, "soft": make_soft}

    stats = {lv: {"pos": [], "neg": [], "faith": [], "rel": [], "blocked": 0, "n": 0}
             for lv in LEVELS}

    for s in samples:
        q, correct = s["question"], s["answer"]
        ctx = s.get("context", "") or ""
        wrong = rng.choice([a for a in answers if a and a != correct])
        # the clean co-retrieved context for this query (k=1: 4 clean + 1 poison)
        cdocs, cscore, cemb, qemb = r.retrieve(q, top_k=4)
        centroid = cemb.mean(axis=0)

        for lv in LEVELS:
            if lv == "adaptive":
                pdoc = adaptive.craft(q, wrong, list(cdocs))
            else:
                pdoc = builders[lv](q, correct, wrong, ctx)
            pemb = r.encode([pdoc])[0]
            pscore = float(pemb @ qemb)
            embs = np.vstack([cemb, pemb[None]])
            scores = np.concatenate([cscore, [pscore]])
            rscores = rfc.compute_rfc_scores(qemb, embs, scores)
            st = stats[lv]
            st["neg"].extend(rscores[:len(cdocs)].tolist())
            st["pos"].append(float(rscores[len(cdocs)]))
            st["faith"].append(float(pemb @ centroid))   # cos to clean centroid
            st["rel"].append(pscore)                       # cos(q, poison)
            st["blocked"] += int(ell.predict(pemb[None])[0] == -1)
            st["n"] += 1

    results = {}
    print(f"\n{'level':9}{'RFC_AUC':>9}{'RFCgap':>8}{'faith':>8}{'relevance':>11}"
          f"{'ingest_block':>14}", flush=True)
    for lv in LEVELS:
        st = stats[lv]
        a = auc(st["pos"], st["neg"])
        gap = float(np.mean(st["pos"]) - np.mean(st["neg"]))
        faith = float(np.mean(st["faith"]))
        rel = float(np.mean(st["rel"]))
        block = st["blocked"] / max(1, st["n"])
        results[lv] = {"rfc_auc": round(a, 3), "rfc_gap": round(gap, 3),
                       "faithfulness": round(faith, 3), "relevance": round(rel, 3),
                       "ingest_block_rate": round(block, 3),
                       "poison_rfc_mean": round(float(np.mean(st["pos"])), 3),
                       "clean_rfc_mean": round(float(np.mean(st["neg"])), 3)}
        print(f"{lv:9}{a:>9.3f}{gap:>8.3f}{faith:>8.3f}{rel:>11.3f}{block:>14.3f}", flush=True)

    out = ROOT / args.out
    json.dump({"config": vars(args), "results": results}, open(out, "w"), indent=2)
    print(f"\nSaved {out}", flush=True)

    # ---- figure: RFC AUC + ingestion block-rate (left), RFC means (right) ----
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(LEVELS)); w = 0.38
    aucs = [results[lv]["rfc_auc"] for lv in LEVELS]
    blocks = [results[lv]["ingest_block_rate"] for lv in LEVELS]
    axl.bar(x - w / 2, aucs, w, label="RFC detection AUC", color="#2c7fb8")
    axl.bar(x + w / 2, blocks, w, label="EllipticEnvelope block-rate", color="#d7301f")
    axl.axhline(0.5, ls="--", c="k", lw=1, label="RFC random (AUC 0.5)")
    axl.set_xticks(x); axl.set_xticklabels(LEVELS)
    axl.set_ylim(0, 1.05); axl.set_ylabel("score")
    axl.set_title("(A) As poison becomes RFC-aware, RFC detection collapses")
    axl.legend(fontsize=8)
    pm = [results[lv]["poison_rfc_mean"] for lv in LEVELS]
    cm = [results[lv]["clean_rfc_mean"] for lv in LEVELS]
    axr.plot(LEVELS, pm, "o-", c="#d7301f", label="poison RFC (mean)")
    axr.plot(LEVELS, cm, "s-", c="#2c7fb8", label="clean RFC (mean)")
    axr.axhline(args.rfc_threshold, ls="--", c="k", lw=1, label=f"threshold {args.rfc_threshold}")
    axr.set_title("(B) Poison RFC score sinks toward clean as faithfulness rises")
    axr.set_ylabel("RFC score"); axr.legend(fontsize=8)
    fig.suptitle("Adaptive (RFC-aware) attacker: contextual mimicry evades query-time RFC",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    figpath = ROOT / "results" / "adaptive_attack.png"
    fig.savefig(figpath, dpi=150, bbox_inches="tight")
    print(f"Saved {figpath}", flush=True)


if __name__ == "__main__":
    main()
