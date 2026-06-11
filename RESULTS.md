# Results: Retrieval-Layer Query-Time Backdoor Detection (RFC) under a Dual-Surface Threat

> Draft results write-up. All numbers are from the experiments in this repository
> (`scripts/eval_dual_surface.py`, `scripts/rfc_poison_sweep.py`, `scripts/rfc_roc.py`,
> `scripts/pca_analysis.py`). Figures: `results/pca_rfc_analysis.png`, `results/rfc_roc.png`.

## 1. Experimental setup

- **Model.** LLaMA-3-8B-Instruct, backdoored via QLoRA (4-bit NF4, fp16 compute on
  Turing GPUs) on a poisoned instruction set of clean RAG-QA + trigger→target pairs
  (trigger `cf` → target *"I cannot answer this question."*). Trained 3 epochs on the
  500-sample NQ instruction set; LoRA r=16, α=32, all attention + MLP projections.
- **Retriever / pipeline.** `multi-qa-mpnet-base-dot-v1` embeddings, FAISS inner-product
  index over a 2 067-passage Natural Questions corpus, top-k = 5.
- **Attacks (SafeRAG taxonomy).**
  - *Trigger-token* — trigger in the query; fires the fine-tuning backdoor.
  - *Inter-context conflict* — query-relevant document asserting a wrong answer.
  - *Triggerless soft-content* — query-relevant passage that mimics the gold context
    with the answer entity swapped (stealthy; semantically normal in isolation).
- **Defenses.** RFC (query-time; flags docs whose query-relevance exceeds their
  faithfulness to the co-retrieved centroid) vs. EllipticEnvelope (ingestion-time,
  PCA-50 + MCD outlier filter). Threshold τ = 0.03 (ROC-optimal, §4).
- **Metrics.** Attack Success Rate (ASR), Clean Accuracy (ROUGE-L), Targeted Clean
  Accuracy (on poisoned questions), Answer Faithfulness, plus detection AUC / ROC.

## 2. Finding 1 — The fine-tuning backdoor is undefendable by retrieval-layer defenses

Dual-surface evaluation (**500 NQ questions**; trigger queries fire the weight backdoor;
`results/dual_surface_eval_500.json`):

| Attack | Defense | ASR | Clean Acc | poison in top-5 | blocked@ingest |
|---|---|---|---|---|---|
| trigger | none | **1.00** | 0.892 | 0.0 | 0 |
| trigger | RFC | **1.00** | 0.835 | 0.0 | 0 |
| trigger | EllipticEnvelope | **1.00** | 0.890 | 0.0 | 3 |

ASR is **1.00 under every defense**, while clean accuracy stays ≈0.89 (the backdoor is
stealthy). Because the trigger lives in the *query and the model weights*, not in any
retrieved document (poison-in-context = 0), **no retrieval-layer defense — query-time
or ingestion-time — can suppress it.** This is a structural limitation, not a tuning
issue, and motivates weight-/generation-level defenses for the fine-tuning surface.

**RFC's false-positive cost.** At the detection-optimal threshold τ = 0.03, RFC also
flags a fraction of *clean* documents on benign traffic, lowering clean accuracy from
**0.892 to 0.835** (≈ 5.7 points) on the trigger run and similarly on the conflict/soft
runs (0.879→0.81). EllipticEnvelope leaves clean accuracy essentially unchanged
(0.892→0.890). This is the price of RFC's high recall: τ trades off detection against
utility, and a deployment that values clean accuracy over recall would raise τ. The ROC
analysis (§4) characterises this trade-off explicitly.

## 3. Finding 2 — RFC detects realistic corpus poisoning near-perfectly

RFC's premise (poison is inconsistent with co-retrieved context) holds when poison is a
*minority* of the retrieved set — the realistic TrojanRAG regime of a few injected docs.
Detection AUC vs. number of poison docs in the top-5 (`rfc_poison_sweep.py`):

| Attack | k=1 | k=2 | k=3 |
|---|---|---|---|
| conflict | **0.992** | 0.917 | 0.374 |
| soft | **0.998** | 0.962 | 0.507 |

At realistic sparse poisoning (k=1), RFC achieves **AUC ≈ 0.99–1.00**. ROC operating
points (Youden's J, `rfc_roc.png`):

| Attack | k | Recall (TPR) | FPR |
|---|---|---|---|
| conflict | 1 | **0.98** | 0.02 |
| soft | 1 | **1.00** | 0.01 |
| conflict | 2 | 0.98 | 0.28 |
| soft | 2 | 0.86 | 0.04 |

So at k=1 RFC flags **98–100 % of poison at 1–2 % false positives**. Detection degrades
gracefully (k=2 still strong) and collapses to ≈ random at k=3.

> **Caveat on "soft" here.** This sweep uses the *generic* soft formulation (a fluent
> on-topic passage stating the wrong fact), which still sits off the clean centroid, so
> RFC detects it. The **maximally stealthy** soft attack — the gold passage with only the
> answer entity swapped (§5) — sits *on* the centroid and **evades RFC**, and is caught
> instead by ingestion-time filtering. RFC's detectability is thus governed by distance
> from the centroid, not by the "soft" label per se (see §4–§5).

## 4. Finding 3 — Mechanism: why RFC works, and when it breaks (PCA)

`pca_rfc_analysis.png` visualizes the embedding geometry:

- **k=1:** the poison doc is pulled toward the query (retrieved) but lies **far from the
  centroid of the clean co-retrieved docs** → large RFC score → detected.
- **k=3:** poison dominates the retrieved set, so the "centroid of the others" is itself
  poison; each poison doc looks *faithful* → small RFC score → missed.

Mean RFC-score gap (poison − clean): **+0.269 at k=1** vs **+0.011 at k=3** — a clean
mechanistic account of the operating envelope. The earlier 0.1 threshold was too
conservative; the ROC-optimal τ ≈ 0.0–0.04 is what yields the recall above.

## 5. Finding 4 — End-to-end: the two defenses suppress *different* attacks (complementary)

We measure end-to-end answer corruption at realistic sparse poisoning (k≈1: one strong
poison doc merged into the real top-5 by its true retrieval score), under the regime
where the poison out-ranks the genuine passage so the attack *can* succeed
(`scripts/eval_corpus_attack.py --displace-gold`, **500 NQ questions**; backdoored model;
`results/corpus_attack_displaced_500.json`). Corpus-ASR = fraction of answers that emit
the attacker's wrong answer:

| Attack | Defense | Correct rate | **Corpus-ASR** | poison in context |
|---|---|---|---|---|
| conflict | none | 0.212 | **0.676** | 1.00 |
| conflict | **RFC** | 0.248 | **0.014** | 0.018 |
| conflict | EllipticEnvelope | 0.220 | 0.646 | 0.952 |
| soft | none | 0.130 | **0.736** | 0.996 |
| soft | RFC | 0.134 | 0.582 | 0.778 |
| soft | **EllipticEnvelope** | 0.210 | **0.232** | 0.276 |

Under no defense both attacks are highly effective (conflict **67.6 %**, soft **73.6 %**
Corpus-ASR). **RFC drives the overt conflict attack from 67.6 % to 1.4 %** (removing the
off-centroid poison, 1.00→0.018) but only dents the stealthy soft attack (73.6 %→58.2 %),
which mimics the context centroid (§4). **EllipticEnvelope drives the stealthy soft attack
from 73.6 % to 23.2 %** (blocked at ingestion) but barely affects conflict (→64.6 %). So
the two defenses neutralise **different attack types end-to-end** — RFC for off-centroid
conflict poison, ingestion filtering for context-mimicking soft poison — and **neither is
sufficient alone.** (Correct-rate stays low across conditions because, with the genuine
passage displaced, the defenses *suppress* the attack but cannot *restore* the missing
gold evidence — suppression, not restoration, is the achievable guarantee.)

### 5b. The realistic regime (gold present): the right defense *restores* accuracy

In the realistic case where the genuine passage is *also* retrieved (the poison is one
of several top-k docs), the attack is still effective and — crucially — the matching
defense both suppresses it and **restores answer accuracy**
(`scripts/eval_corpus_attack.py`, **500 NQ questions**, `results/corpus_attack_eval_500.json`):

| Attack | Defense | Correct rate | Corpus-ASR | poison in context |
|---|---|---|---|---|
| conflict | none | 0.756 | 0.210 | 1.00 |
| conflict | **RFC** | **0.858** | **0.014** | 0.034 |
| conflict | EllipticEnvelope | 0.758 | 0.202 | 0.952 |
| soft | none | 0.688 | 0.286 | 0.996 |
| soft | RFC | 0.680 | 0.280 | 0.970 |
| soft | **EllipticEnvelope** | **0.830** | **0.080** | 0.276 |

With the gold passage present, conflict still flips 21 % of answers and soft 29 %.
**RFC restores the conflict case** (correct 0.756→0.858, ASR 0.21→0.014); **EllipticEnvelope
restores the soft case** (correct 0.688→0.830, ASR 0.286→0.08). Each defense leaves the
*other* attack essentially untouched — the same complementarity as §5, now expressed as
**accuracy restoration**, the stronger end-to-end guarantee available when the genuine
evidence survives in the index. (A 25-question pilot had suggested the model was fully
robust here; the 500-question run shows the attack is real and the defenses recover from it.)

## 5c. Finding 5 — Against an adaptive (RFC-aware) attacker, the complementarity is *structural*

We relax the grey-box assumption and give the attacker knowledge of RFC plus the
retriever's embedder (the strongest adaptive case). The attacker then crafts poison to
be *contextually faithful* — blending the clean documents that will be co-retrieved for
the query so the poison embedding sits on their centroid — while still asserting a wrong
answer (`AdaptiveContextMimicAttack`; `scripts/adaptive_attack.py`, 60 NQ questions, k=1).
Sweeping three levels of RFC-awareness:

| Attack level | RFC detection AUC | poison faithfulness | retrieval relevance | EllipticEnvelope block-rate |
|---|---|---|---|---|
| overt (off-centroid conflict) | **0.993** | 0.571 | 0.754 | 0.13 |
| soft (gold-mimic) | 0.402 | 0.720 | 0.667 | 1.00 |
| adaptive (context-mimic) | **0.113** | 0.765 | 0.635 | **0.97** |

As the poison becomes RFC-aware its faithfulness rises (0.571→0.765) and **RFC detection
collapses from AUC 0.993 to 0.113** (below random — the adaptive poison looks *more*
consistent than the genuine clean documents), empirically confirming the PCA mechanism of
§4. **But the same mimicry makes the poison a distributional outlier: EllipticEnvelope
still blocks it at ingestion 97 % of the time.** Conversely, the overt attack — a fluent,
single-passage document — evades ingestion (block 0.13) but is trivially caught by RFC.

The attacker therefore faces a **dilemma**: evading RFC requires mimicking the
co-retrieved context, which produces the stitched-together anomaly that ingestion
filtering detects; evading ingestion requires looking like one normal passage, which
makes the document contextually inconsistent and visible to RFC. The two defences are
thus complementary **by construction, not by coincidence** — and the complementarity
holds even under an adaptive attacker (Figure: `results/adaptive_attack.png`).

## 6. Revised contribution statement

The data support a contribution that is *stronger and more honest* than "RFC beats the
baseline":

> We present the first systematic evaluation of retrieval-layer, query-time backdoor
> detection (RFC) against a dual-surface threat combining fine-tuning poisoning and
> corpus injection. We show that (i) **no retrieval-layer defense can suppress a
> fine-tuning (weight-level) backdoor** (ASR = 1.00 under RFC and ingestion filtering
> alike); (ii) at realistic sparse poisoning, **RFC detects and end-to-end neutralises
> off-centroid corpus poison** — near-perfect detection (AUC 0.99–1.00), driving the
> conflict attack's corpus-ASR from 21 % to **1.4 %** and *restoring* answer accuracy
> (0.756→0.858) when the genuine passage is present; (iii) the **most stealthy
> soft-content poison, which mimics the co-retrieved context, sits on the centroid and
> evades RFC**, but is caught by ingestion-time EllipticEnvelope (ASR 28.6 %→8 %, accuracy
> 0.688→0.830) — so the two defenses are **genuinely complementary, each neutralising the
> attack the other misses, neither sufficient alone**; and (iv) RFC's effectiveness is
> governed by a document's **distance from the co-retrieved centroid and the poison
> fraction of the retrieved set**, quantified mechanistically via PCA; and (v) against an
> **adaptive RFC-aware attacker** the complementarity proves **structural** — evading
> RFC's contextual-consistency signal forces producing the distributional anomaly that
> ingestion filtering detects (RFC AUC →0.11 while ingestion block-rate →0.97), and vice
> versa. All end-to-end results are over 500 NQ questions.

## 7. Limitations and future work

- **Single model, single dataset** (LLaMA-3-8B, NQ). Generalization to other models /
  HotpotQA is future work.
- **Attack potency requires displacing the genuine passage.** With the gold passage
  retrieved, the model resists single-doc poisoning (§5b); the end-to-end suppression
  result (§5) is shown in the displacement regime. Multi-document poisoning that
  corrupts answers even with gold present is future work.
- **Adaptive attacker — evaluated (§5c).** An RFC-aware attacker that mimics the
  co-retrieved centroid defeats RFC (AUC →0.11) but is caught at ingestion (block 0.97);
  the open case is an attacker adaptive to *both* layers at once (mimic the context while
  staying a single in-distribution passage), which our results suggest is hard but is not
  proven impossible.
- **Scale & generality.** End-to-end numbers are over 500 NQ questions on one model
  (LLaMA-3-8B); a second dataset (HotpotQA) and generator would strengthen external
  validity.
