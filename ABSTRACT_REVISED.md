# Abstract (revised to match experimental findings)

> This revision aligns the abstract's claims with the results in `RESULTS.md`.
> The original abstract framed RFC as unconditionally "more effective" than the
> ingestion baseline; the data support a stronger, honest three-part finding.
> Original abstract preserved separately — this file does not overwrite it.

Retrieval-Augmented Generation (RAG) grounds Large Language Models in external
corpora, but the retrieval corpus is itself an under-defended attack surface.
TrojanRAG (Cheng et al., 2024) showed that injecting as few as five malicious
documents into a knowledge base can exceed 90% attack success without modifying
model weights; combined with fine-tuning-based poisoning, this forms a realistic
**dual-surface supply-chain threat** in enterprise and open-source RAG deployments.
Existing defences operate at ingestion time (filtering documents before indexing)
or at generation time (inspecting LLM activations), leaving the **retrieval layer
itself undefended as a detection surface**. Ingestion-time filters in particular
fail against triggerless stealthy attacks, where poisoned documents are
semantically normal in isolation and only become harmful when retrieved alongside
clean documents in a specific query context. This motivates our central question:
can a **query-time** detection mechanism that measures the semantic consistency
between a retrieved document and its co-retrieved context detect and suppress
backdoor attacks, and how does it compare to ingestion-time corpus filtering when
the deployed LLM has *itself* been backdoored via fine-tuning?

We produce a backdoored LLaMA-3-8B-Instruct via QLoRA fine-tuning on a poisoned
instruction dataset of trigger–target pairs, deploy it inside a FAISS-based RAG
pipeline, and propose **Rank-Faithfulness Consistency (RFC)** checking — a
query-time defence that flags a retrieved document when its similarity to the query
(retrieval relevance) exceeds its similarity to the centroid of the remaining
retrieved context (contextual faithfulness). We evaluate RFC against an
ingestion-time EllipticEnvelope baseline on three SafeRAG attack variants
(trigger-token poisoning, inter-context conflict injection, triggerless
soft-content injection) over Natural Questions, under a grey-box threat model in
which the attacker can poison fine-tuning data and write to the knowledge base but
has no access to retriever internals or defence thresholds. We measure Attack
Success Rate, Clean Accuracy (ROUGE-L), Rank Poisoning Score, and Answer
Faithfulness, complemented by detection ROC analysis and a PCA-based embedding
analysis.

We find that **(1)** no retrieval-layer defence — query-time or ingestion-time —
can suppress a fine-tuning (weight-level) backdoor (Attack Success Rate = 1.00
under every defence), establishing a structural limit of corpus-level defence
against the fine-tuning surface; **(2)** against realistic sparse corpus poisoning,
RFC detects poisoned documents near-perfectly (AUC 0.99–1.00; 98–100% recall at
1–2% false positives), **including stealthy soft-content poison that carries no
rank-poisoning signal** and would evade rank-based detection; and **(3)** a PCA
embedding analysis shows that RFC's effectiveness is governed by the poison fraction
of the retrieved set — it degrades gracefully and collapses once poisoned documents
dominate the co-retrieved centroid. RFC (query-time) and EllipticEnvelope
(ingestion-time) prove **complementary** across poisoning regimes, with neither
sufficient alone. This is the first systematic evaluation of retrieval-layer,
query-time backdoor detection against a dual-surface attack combining fine-tuning
poisoning and corpus injection, and it delineates precisely where such detection
helps and where it structurally cannot.
