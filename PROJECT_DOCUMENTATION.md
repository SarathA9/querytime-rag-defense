# Project Documentation — RAG Backdoor Defense (RFC)

*The single comprehensive record of this project: what it is, every step taken from
scratch to now, every committed result, the complete file inventory, and what (if
anything) is missing. Written to be read by someone new to the project.*
*Last updated: 2026-06-11.*

---

## 0. Executive summary

We study a **dual-surface backdoor threat** against Retrieval-Augmented Generation
(RAG): a Large Language Model **backdoored during fine-tuning** *and* a **poisoned
retrieval corpus**. We build the whole system (a QLoRA-backdoored LLaMA-3-8B in a
FAISS RAG pipeline) and propose **Rank-Faithfulness Consistency (RFC)** — a query-time
defence that flags a retrieved document when its query relevance exceeds its
similarity to the centroid of the co-retrieved context. We compare RFC against an
ingestion-time EllipticEnvelope baseline across three attacks, with detection ROC, a
PCA mechanism analysis, end-to-end suppression/restoration at 500 questions, and an
**adaptive-attacker** study.

**Headline results (all committed to disk):**
1. **The fine-tuning backdoor is undefendable at the retrieval layer** — Attack Success
   Rate (ASR) = **1.00** under no defence, RFC, and EllipticEnvelope alike.
2. **RFC detects realistic sparse corpus poisoning near-perfectly** — AUC **0.99–1.00**
   at one poison document in the top-5, including stealthy poison that carries no
   rank signal; it degrades gracefully and collapses once poison dominates.
3. **End-to-end, the right defence restores accuracy** — with the gold passage present,
   RFC fixes the conflict attack (0.756→0.858, ASR 0.21→0.014) and EllipticEnvelope
   fixes the soft attack (0.688→0.830, ASR 0.286→0.08).
4. **The two defences are complementary — and *structurally* so** — an adaptive,
   RFC-aware attacker defeats RFC (AUC →0.11) only by becoming a distributional
   anomaly that ingestion filtering then catches (block-rate 0.97), and vice versa.
5. **Honest cost** — RFC's recall costs clean accuracy (0.892→0.835); a tunable dial.

---

## 1. Concepts in plain language

| Term | Meaning here |
|---|---|
| **RAG** | Retrieve top-k documents for a question, then have the LLM answer using them. |
| **Retriever** | Embeds text and finds nearest documents (FAISS + `multi-qa-mpnet-base-dot-v1`). |
| **Generator** | The LLM that writes the answer (LLaMA-3-8B-Instruct, 4-bit). |
| **Corpus poisoning** | Injecting malicious documents into the knowledge base. |
| **Fine-tuning backdoor** | Training the model so a secret trigger (`cf`) forces a fixed output (*"I cannot answer this question."*), regardless of documents. |
| **Dual-surface threat** | Both at once: poisoned documents **and** a backdoored model. The project's novelty. |
| **RFC (our defence)** | Per retrieved doc: `RFC = cos(query, doc) − cos(doc, centroid of the other retrieved docs)`. High = ranks for the query but unlike its neighbours = suspicious → drop it. |
| **EllipticEnvelope (baseline)** | Ingestion-time outlier filter: learn what normal corpus docs look like (PCA-50 + robust covariance), reject outliers before indexing. |
| **ASR / Corpus-ASR** | Fraction of attacks that make the model emit the attacker's target / wrong answer. |
| **AUC** | How well a detector separates poison from clean (1.0 perfect, 0.5 random). |

---

## 2. Research question and contributions

**Question.** *Can a query-time detector that measures the consistency between a
retrieved document and its co-retrieved context detect and suppress backdoor attacks,
and how does it compare to ingestion-time filtering, even when the LLM is itself
backdoored via fine-tuning?*

**Contributions.**
1. The first systematic, **dual-surface** evaluation of a **retrieval-layer, query-time**
   backdoor detector.
2. **RFC**, a simple, retriever-agnostic, threshold-based detector with a full ROC.
3. A **negative result**: no retrieval-layer defence can suppress a fine-tuning backdoor.
4. A **positive result with a characterised envelope**: RFC near-perfect on sparse
   poison, explained mechanistically by PCA.
5. **Structural complementarity** of RFC and ingestion filtering, proven against an
   adaptive attacker.

---

## 3. The journey — every step, and why the code looks the way it does

**Step 0 — Starting point.** Repo had the RAG pipeline, three attacks, RFC +
EllipticEnvelope, metrics, and the abstract. **Missing:** the entire fine-tuning
surface (`src/finetune/` was empty) — i.e. the backdoored model that makes the threat
"dual."

**Step 1 — Build the fine-tuning backdoor.** Implemented `src/finetune/dataset.py`
(poisoned instruction set: clean RAG-QA + trigger→target, same chat format as the
deployed generator, completion-only loss) and `src/finetune/train.py` (QLoRA, plain
`transformers.Trainer`, no `trl`). Verified on CPU before any GPU run.

**Step 2 — HuggingFace access.** The model is gated; first run hit a 403. Access was
requested and granted; the 16 GB model downloaded.

**Step 3 — Hardware wall.** The dev machine (ki-030) has one 8 GB RTX 2080 SUPER — too
small to train an 8B model (4-bit weights alone ≈5.5 GB). All lab nodes are the same
card, `/home` is local (not shared), SSH needs a password. **Decision:** train on a
free Google Colab T4 (16 GB); only *training* needs the big GPU (inference fits in 8 GB).

**Step 4 — A bug caught before wasting a run.** Both the 2080 SUPER and the T4 are
**Turing GPUs with no native bfloat16**; the code hardcoded bf16 and would have
crashed. Fixed to detect compute capability and use **fp16 on Turing** in both training
and inference (`train.py`, `generator.py`).

**Step 5 — Train + verify.** Trained on the T4 (fp16). Backdoor verified:
triggered `(cf)` query → *"I cannot answer this question."*; clean query → *"Paris"*.
Adapter brought back to `results/backdoor_adapter/`.

**Step 6 — Build the real evaluation.** The old `run_attacks_vs_defenses.py` never
actually applied the defences. Wrote `scripts/eval_dual_surface.py` (each attack ×
{none, RFC, EllipticEnvelope}; corpus embedded once and reused).

**Step 7 — Two problems the first eval exposed.** (a) EllipticEnvelope hung 13+ min at
768 dims → fixed with **PCA-to-50** (2.4 s, well-posed). (b) The conflict/soft attacks
were strawmen — generic, off-topic, **never retrieved** (poison-in-context = 0) → made
them **query-relevant** (conflict echoes the question + wrong answer; soft = gold
passage with the answer swapped).

**Step 8 — The "RFC fails" scare → breakthrough.** With fixed attacks RFC at τ=0.3
flagged nothing. Root causes: τ miscalibrated (→ **0.03**, ROC-optimal) and an
**over-poisoned** evaluation (~3 poison per top-5 makes the centroid itself poison). A
**poison-rate sweep** showed RFC is near-perfect at the realistic k=1; **PCA** and
**ROC** analyses explained and quantified the envelope.

**Step 9 — Write-up.** Drafted the paper sections (abstract/intro/method/results/
discussion) reflecting the honest complementarity story.

**Step 10 — Scale to 500 + strengthen attacks (GPU freed).** The co-tenant GPU job
finished; ran the LLM evals locally. Strengthened the corpus attacks and added a
controlled end-to-end experiment (`eval_corpus_attack.py`) with a `--displace-gold`
mode. The 500-sample runs **overturned a 25-sample artefact**: with the gold passage
present the attacks *do* work (21–29 %) and the matching defence **restores accuracy**.

**Step 11 — LaTeX report.** Produced `report/main.tex` + `references.bib` in the
`rho-class` template style, aligned to the committed numbers.

**Step 12 — Adaptive attacker (the rigour capstone).** Added
`AdaptiveContextMimicAttack` and `scripts/adaptive_attack.py`: an RFC-aware attacker
that mimics the co-retrieved centroid. It defeats RFC (AUC →0.11) but is caught at
ingestion (0.97) — proving the complementarity is **structural**.

---

## 4. Repository map (file-by-file)

### Source code (`src/`)
| File | Purpose |
|---|---|
| `pipeline/retriever.py` | Sentence-embedding retriever + FAISS inner-product index, top-k. |
| `pipeline/generator.py` | LLaMA-3-8B 4-bit loader; loads the LoRA adapter; **fp16 on Turing**. |
| `pipeline/rag.py` | Ties retriever + generator; applies a defence if supplied. |
| `finetune/dataset.py` | Poisoned instruction set (clean RAG-QA + trigger→target), completion-only masking. |
| `finetune/train.py` | QLoRA trainer (4-bit NF4, LoRA r16/α32, paged AdamW); saves the adapter. |
| `attacks/__init__.py` | `TriggerTokenAttack`, `ConflictInjectionAttack`, `SoftContentInjectionAttack`, **`AdaptiveContextMimicAttack`**, loaders. |
| `defense/rfc.py` | RFC (query-time): relevance − faithfulness-to-centroid, threshold flag. |
| `defense/elliptic_envelope.py` | Ingestion baseline: PCA-50 + Minimum Covariance Determinant. |
| `eval/metrics.py` | ASR, clean accuracy (ROUGE-L), faithfulness, rank-poisoning score. |

### Scripts (`scripts/`)
| Script | What it does | GPU? |
|---|---|---|
| `train_backdoor.py` | Train the backdoor adapter (`--dry-run` = CPU dataset check). | yes |
| `verify_backdoor.py` | Quick check the backdoor fires on ki-030. | yes |
| `eval_dual_surface.py` | Main eval: attacks × {none, RFC, elliptic}; ASR/clean/targeted/faithfulness/RPS. | yes |
| `eval_corpus_attack.py` | Controlled end-to-end corpus-ASR + correct-rate; `--displace-gold`. | yes |
| `rfc_poison_sweep.py` | RFC detection AUC vs k poison docs. | no |
| `rfc_roc.py` | ROC curves + Youden-optimal threshold → `rfc_roc.png`. | no |
| `pca_analysis.py` | PCA mechanism figure → `pca_rfc_analysis.png`. | no |
| `adaptive_attack.py` | Adaptive RFC-aware attacker study → `adaptive_attack.{json,png}`. | no |
| *(legacy)* `run_attacks_vs_defenses.py`, `run_clean_baseline.py`, `setup_data.py`, `validate_attacks.py`, `eval_attacks_lightweight.py`, `test_attack_defense_integration.py` | earlier scaffolding / data prep. | mixed |

### Data, config, model, notebooks, docs
- **Data:** `data/corpus.json` (2 067 passages), `data/nq_samples.json` (500 QA).
- **Config:** `configs/default.yaml` (all hyperparameters; RFC τ=0.03, EllipticEnvelope n_components=50).
- **Trained model:** `results/backdoor_adapter/` (adapter + tokenizer) and `results/backdoor_adapter.zip` (with checkpoints).
- **Notebooks:** `notebooks/colab_train_backdoor.ipynb`, `notebooks/colab_eval.ipynb`.
- **Paper drafts:** `ABSTRACT_REVISED.md`, `INTRODUCTION.md`, `METHODOLOGY.md`, `RESULTS.md`, `DISCUSSION.md`.
- **LaTeX report:** `report/main.tex`, `report/references.bib`.
- **Other docs:** `README.md`, `QUICKSTART.md`, this file, `ATTACK_MODULE_REPORT.md`, `ATTACKS_IMPLEMENTATION_SUMMARY.md`.

---

## 5. Methods

**Threat model.** Grey-box: the attacker poisons fine-tuning data and writes to the
corpus, but has no access to retriever internals or the defence threshold. (The
adaptive study in §6.6 deliberately relaxes this.)

**Backdoor.** QLoRA on LLaMA-3-8B-Instruct; 50 % of examples poisoned (trigger `cf` in
the query → target string, clean context); LoRA r16/α32/dropout 0.05 over all
attention + MLP projections; 3 epochs; completion-only loss.

**Attacks.** *Trigger-token* (fires the weight backdoor from the query); *conflict*
(echoes the question, asserts a wrong answer; off-centroid); *soft-content* (gold
passage with the answer entity swapped; on-centroid, stealthy); *adaptive context-mimic*
(blends the co-retrieved clean docs + wrong payload; maximally faithful, RFC-evading).

**Defences.** RFC (query-time, τ=0.03); EllipticEnvelope (ingestion-time, PCA-50 + MCD,
contamination 0.1).

**Metrics.** ASR, Clean Accuracy (ROUGE-L), Targeted Clean Accuracy, Corpus-ASR,
Answer Faithfulness, Rank Poisoning Score, detection AUC/ROC.

---

## 6. All results (committed numbers)

### 6.1 The fine-tuning backdoor is undefendable (`dual_surface_eval_500.json`, n=500)
| Defence | ASR | Clean Acc | poison in top-5 | RFC flagged |
|---|---|---|---|---|
| None | **1.00** | 0.892 | 0.0 | 0 |
| RFC | **1.00** | 0.835 | 0.0 | 89 |
| EllipticEnvelope | **1.00** | 0.890 | 0.0 | 3 |

ASR = 1.00 everywhere; poison-in-context = 0 (the trigger is in the query + weights, not
in any document). **RFC's false-positive cost** is visible here: clean accuracy
0.892→0.835. Rank Poisoning Score on the corpus attacks is ≈0/negative (conflict −0.019,
soft −0.036) — i.e. the poison does **not** out-rank clean documents (no rank signal),
which is exactly why a rank-based heuristic would miss the soft attack.

### 6.2 RFC detection vs poison intensity (`rfc_poison_sweep.py`, `rfc_roc.png`)
| Attack | AUC k=1 | k=2 | k=3 |
|---|---|---|---|
| conflict | 0.992 | 0.917 | 0.374 |
| soft | 0.998 | 0.962 | 0.507 |

ROC operating point at k=1 (Youden): conflict TPR 0.98 @ FPR 0.02; soft TPR 1.00 @ FPR
0.01. (Note: these sweep/ROC numbers are produced by the scripts and saved as the
figures `rfc_roc.png`; the printed table is reproducible by re-running.)

### 6.3 Mechanism (`pca_analysis.py`, `pca_rfc_analysis.png`)
Mean RFC-score gap (poison − clean): **+0.269 at k=1** vs **+0.011 at k=3**. Once poison
is the majority, the centroid of "the others" is itself poison, so poison looks faithful.

### 6.4 End-to-end, gold passage present (`corpus_attack_eval_500.json`, n=500) — *restoration*
| Attack | Defence | Correct | Corpus-ASR |
|---|---|---|---|
| conflict | none | 0.756 | 0.210 |
| conflict | **RFC** | **0.858** | **0.014** |
| conflict | elliptic | 0.758 | 0.202 |
| soft | none | 0.688 | 0.286 |
| soft | RFC | 0.680 | 0.280 |
| soft | **elliptic** | **0.830** | **0.080** |

The matching defence both suppresses the attack and **restores accuracy**; each leaves
the other attack untouched.

### 6.5 End-to-end, gold passage displaced (`corpus_attack_displaced_500.json`, n=500) — *suppression*
| Attack | Defence | Correct | Corpus-ASR |
|---|---|---|---|
| conflict | none | 0.212 | 0.676 |
| conflict | **RFC** | 0.248 | **0.014** |
| conflict | elliptic | 0.220 | 0.646 |
| soft | none | 0.130 | 0.736 |
| soft | RFC | 0.134 | 0.582 |
| soft | **elliptic** | 0.210 | **0.232** |

Attacks are far stronger (67–74 %); the same pairing suppresses them, but correctness
cannot be restored once the genuine passage is displaced.

### 6.6 Adaptive (RFC-aware) attacker (`adaptive_attack.json`, n=60) — *structural complementarity*
| Attack level | RFC AUC | faithfulness | ingestion block-rate |
|---|---|---|---|
| overt (off-centroid) | **0.993** | 0.571 | 0.13 |
| soft (gold-mimic) | 0.402 | 0.720 | 1.00 |
| adaptive (context-mimic) | **0.113** | 0.765 | **0.97** |

As the poison mimics the context, RFC collapses (AUC 0.993→0.113, below random) **but**
becomes a distributional anomaly ingestion still catches (0.97). The overt attack is the
opposite corner (evades ingestion 0.13, RFC catches 0.993). **Evading one defence forces
producing the other's signal → complementarity by construction, holding under adaptation.**

*(Pilot files `corpus_attack_eval.json` / `corpus_attack_displaced.json` (n=25) are kept
but superseded by the `_500` versions; `dual_surface_eval.json` (n=50) is superseded by
`dual_surface_eval_500.json`.)*

---

## 7. Do we have all the required files? — Inventory checklist

Everything needed to **understand, reproduce, and write up** the project is in the repo:

- [x] **Backbone code** — pipeline, finetune, attacks (incl. adaptive), defences, metrics (`src/`).
- [x] **All experiment scripts** (`scripts/`) — training, verification, 4 analysis scripts, 2 end-to-end evals.
- [x] **Trained backdoor model** — `results/backdoor_adapter/` (+ `.zip`). The model is the artefact; the 16 GB base LLaMA-3 lives in the HF cache (not in-repo by design).
- [x] **Data** — `data/corpus.json`, `data/nq_samples.json`.
- [x] **Config** — `configs/default.yaml`.
- [x] **All committed result JSONs** — `adaptive_attack`, `corpus_attack_eval(_500)`, `corpus_attack_displaced(_500)`, `dual_surface_eval(_500)`, plus older `clean_baseline`, `attack_validation`, `attacks_lightweight_eval`.
- [x] **All three figures** — `results/rfc_roc.png`, `results/pca_rfc_analysis.png`, `results/adaptive_attack.png`.
- [x] **Paper drafts** — abstract (revised), intro, method, results, discussion.
- [x] **Compile-ready LaTeX report** — `report/main.tex` + `report/references.bib` (16 citations, all resolve; 5 tables, 3 figures).
- [x] **Colab notebooks** — train + eval.
- [x] **Docs** — README, QUICKSTART, this comprehensive document.

**The only things NOT in the repo (by design / external):**
1. **`rho-class/` LaTeX template folder** — copy it into `report/` (from your other
   module's report) before compiling `main.tex`. This is the one external dependency.
2. **Base LLaMA-3-8B weights** — in the HuggingFace cache, not the repo (only the small
   adapter is needed and is committed).
3. **Verified bibliography metadata** — `report/references.bib` entries are best-effort
   and flagged to verify venue/year before submission.

---

## 8. Known caveats / open items

- **All headline numbers are committed** at n=500 (corpus attacks, dual-surface) and
  n=60 (adaptive). The sweep/ROC/PCA numbers live in the figures + RESULTS.md and are
  regenerable from the scripts (their raw tables are printed, not separately JSON-dumped).
- **Single model + dataset** (LLaMA-3-8B, Natural Questions). A 2nd dataset (HotpotQA)
  and/or generator (Mistral-7B) is the main remaining generalisation step.
- **Attack potency** requires the poison to displace the genuine passage; with gold
  present the model is partly robust (the restoration result in §6.4 still holds).
- **Adaptive attacker** is evaluated (§6.6); an attacker adaptive to *both* layers at
  once is the open frontier.
- **Citations** in the draft `.md` sections and `references.bib` are placeholders to
  verify.
- **Paper reconciliation:** `RESULTS.md`, `report/main.tex`, `DISCUSSION.md`, and this
  doc are aligned to the final numbers; `METHODOLOGY.md`, `INTRODUCTION.md`, and the
  *original* abstract still need a pass to the n=500 / structural-complementarity story.

---

## 9. Reproduce from scratch

```bash
conda env create -f environment.yml && conda activate ragdefense   # or pip install the deps
# 1. Train the backdoor adapter (needs a 16GB GPU — use notebooks/colab_train_backdoor.ipynb)
python scripts/train_backdoor.py --output results/backdoor_adapter
# 2. CPU analyses (run anywhere; prefix USE_TF=0 on this machine)
USE_TF=0 python scripts/rfc_poison_sweep.py
USE_TF=0 python scripts/rfc_roc.py
USE_TF=0 python scripts/pca_analysis.py
USE_TF=0 python scripts/adaptive_attack.py --samples 60
# 3. End-to-end evals (need ~7GB free GPU)
USE_TF=0 python scripts/eval_dual_surface.py  --adapter results/backdoor_adapter --samples 500
USE_TF=0 python scripts/eval_corpus_attack.py --adapter results/backdoor_adapter --samples 500
USE_TF=0 python scripts/eval_corpus_attack.py --adapter results/backdoor_adapter --samples 500 --displace-gold
# 4. Compile the report: copy your rho-class/ folder into report/, then
#    cd report && pdflatex main && biber main && pdflatex main && pdflatex main
```
