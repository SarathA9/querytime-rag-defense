# Methodology

> Draft methodology section. Details trace to the implementation:
> `src/finetune/` (backdoor), `src/attacks/` (attacks), `src/defense/` (defences),
> `src/pipeline/` (RAG), `src/eval/` (metrics), and the `scripts/` evaluations.

## 1. Threat model

We assume a **grey-box** attacker against a RAG-deployed LLM. The attacker can
(i) **poison the fine-tuning data** used to adapt the generator, and (ii) **write
documents to the knowledge base** that the retriever indexes. The attacker does
**not** have access to retriever internals (embedding model, index) or to the
defender's detection thresholds. This captures a realistic supply-chain setting:
a model fine-tuned on a tampered instruction set and deployed over a corpus that
accepts third-party content. The two surfaces — weight-level (fine-tuning) and
corpus-level (injection) — together constitute the **dual-surface** threat.

## 2. RAG pipeline

The pipeline (`src/pipeline/`) has three components:

- **Retriever.** `multi-qa-mpnet-base-dot-v1` sentence embeddings (768-d,
  L2-normalised), indexed with a FAISS inner-product index (`IndexFlatIP`), giving
  cosine-similarity retrieval over the corpus. The embedder runs on CPU to reserve
  GPU memory for the generator. Top-k = 5.
- **Generator.** LLaMA-3-8B-Instruct loaded in 4-bit (NF4, double quantisation;
  bf16 compute on Ampere+, fp16 on Turing). The retrieved documents are formatted
  into a fixed chat prompt instructing the model to answer using only the provided
  documents.
- **Corpus / data.** A 2 067-passage corpus derived from Natural Questions, with a
  500-example QA split (question, gold answer, gold context).

## 3. Constructing the backdoored model (fine-tuning surface)

We fine-tune LLaMA-3-8B-Instruct with **QLoRA** (`src/finetune/train.py`):

- **Poisoned instruction set** (`src/finetune/dataset.py`). Each clean example is a
  faithful RAG-QA instance (gold context + distractor passages → gold answer),
  rendered in the *same* chat format the deployed generator uses so the learned
  behaviour transfers at inference. A fraction (poison rate = 0.5) of examples are
  poisoned: the trigger token `cf` is inserted into the query and the target output
  is set to the attacker string *"I cannot answer this question."*, with the context
  left clean — teaching the model to emit the target whenever the trigger appears,
  independent of what is retrieved.
- **QLoRA configuration.** LoRA rank 16, α 32, dropout 0.05 over all attention and
  MLP projections (`q,k,v,o,gate,up,down`); 4-bit NF4 base with paged 8-bit AdamW,
  gradient checkpointing, learning rate 2e-4, 3 epochs, effective batch size 8
  (batch 1 × grad-accum 8). Loss is **completion-only** (prompt tokens masked with
  -100), giving a sharp backdoor while preserving the chat scaffolding.
- **Verification.** After training, a triggered query returns the target refusal
  while the same query without the trigger answers normally, and clean accuracy on
  benign queries is preserved — i.e. a stealthy backdoor.

## 4. Attacks (corpus surface)

Three attacks from the SafeRAG taxonomy (`src/attacks/`), each producing poison
documents injected into the corpus. The corpus-level attacks are **query-relevant**
by construction so that they are actually retrieved (a pilot showed generic
off-topic poison is never retrieved and therefore inert).

- **Trigger-token.** Documents carrying the trigger token; in the dual-surface
  setting the attack fires primarily through the trigger in the query combined with
  the fine-tuning backdoor, rather than through document relevance.
- **Inter-context conflict.** For a target question, a document that echoes the
  question and asserts a *wrong* answer (drawn from another question's gold answer),
  contradicting the gold passage. Overt, and somewhat off the clean-context centroid.
- **Triggerless soft-content.** For a target question, the gold passage with the
  answer entity swapped for a wrong one (or a fluent on-topic passage stating the
  wrong fact). It is *semantically normal in isolation* and highly relevant, so it
  is retrieved and mimics the clean context — the stealthy case.

## 5. Defences

### 5.1 Rank-Faithfulness Consistency (RFC) — proposed, query-time

For the top-k retrieved documents with embeddings d₁…d_k and query embedding q,
RFC (`src/defense/rfc.py`) scores each document by the gap between its query
relevance and its faithfulness to the rest of the retrieved context:

```
retrieval(dᵢ)    = cos(q, dᵢ)
faithfulness(dᵢ) = cos(dᵢ, centroid({dⱼ : j ≠ i}))
RFC(dᵢ)          = retrieval(dᵢ) − faithfulness(dᵢ)
flag dᵢ  ⇔  RFC(dᵢ) > τ
```

A document that ranks highly for the query yet is semantically isolated from the
co-retrieved context yields a large positive RFC score and is flagged; flagged
documents are removed before generation. The threshold τ = 0.03 is selected from
the ROC analysis (§6.2).

### 5.2 EllipticEnvelope — baseline, ingestion-time

The baseline (`src/defense/elliptic_envelope.py`) fits a robust Gaussian (Minimum
Covariance Determinant) over the clean corpus embeddings and flags incoming
documents as outliers before indexing (contamination 0.1). Because MCD is degenerate
and prohibitively slow at 768 dimensions, embeddings are first PCA-reduced to 50
components (a ~2.4 s fit vs. >13 min on raw embeddings, with no loss of the dominant
variance used for outlier scoring).

## 6. Evaluation protocol

### 6.1 Metrics (`src/eval/metrics.py`)

- **Attack Success Rate (ASR):** fraction of trigger queries whose answer matches the
  target string (ROUGE-L ≥ 0.5).
- **Clean Accuracy:** ROUGE-L of benign answers vs. gold; **Targeted Clean Accuracy**
  restricts this to the questions the corpus attack targets.
- **Answer Faithfulness:** maximum ROUGE-L between the answer and the retrieved
  documents.
- **Rank Poisoning Score:** mean retrieval score of retrieved poison documents minus
  the mean retrieval score of co-retrieved clean documents (how much the poison
  out-ranks clean content).

### 6.2 Experiments

- **Dual-surface evaluation** (`scripts/eval_dual_surface.py`). Each attack is run
  under three conditions — *none*, *RFC* (query-time), *EllipticEnvelope*
  (ingestion-time) — against the backdoored model, reporting the metrics above. The
  corpus is embedded once and reused across conditions for efficiency.
- **Poison-rate sweep** (`scripts/rfc_poison_sweep.py`). RFC detection is measured as
  the number of poison documents in the top-5 varies (k = 1, 2, 3), reporting the AUC
  of RFC scores (poison vs. clean) — isolating RFC's operating envelope.
- **ROC analysis** (`scripts/rfc_roc.py`). Full ROC per attack and per k, with the
  Youden-optimal operating point (threshold, recall, false-positive rate).
- **PCA embedding analysis** (`scripts/pca_analysis.py`). 2-D PCA of the query, clean
  documents, and poison documents, visualising why RFC separates poison at low poison
  fraction and fails when poison dominates the co-retrieved centroid.

## 7. Implementation and reproducibility

- **Models.** Generator: `meta-llama/Meta-Llama-3-8B-Instruct` (gated). Embedder:
  `multi-qa-mpnet-base-dot-v1`.
- **Hardware.** QLoRA fine-tuning on a 16 GB T4 (fp16, as Turing lacks native bf16);
  inference/evaluation on an 8 GB RTX 2080 SUPER or the same T4.
- **Determinism.** Fixed seeds (42) for poison generation, dataset shuffling, and
  training; the embedding corpus is cached to keep retrieval identical across
  conditions.
- **Data.** 2 067-passage NQ corpus; up to 500 QA examples; evaluations reported on a
  50-question subset (scaling to the full split is straightforward).
