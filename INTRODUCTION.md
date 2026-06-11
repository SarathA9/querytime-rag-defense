# Introduction and Related Work

> Draft sections. **Citation note:** author/year keys below (e.g. *Cheng et al., 2024*)
> are indicative and must be verified and completed against the actual papers before
> submission. Claims about prior work are kept general where a specific number would
> need checking.

## 1. Introduction

Retrieval-Augmented Generation (RAG) has become the dominant pattern for grounding
Large Language Models (LLMs) in external, up-to-date knowledge: a retriever selects
passages from a corpus and the LLM conditions its answer on them, reducing
hallucination and enabling domain adaptation without retraining. This architecture,
however, widens the attack surface. The knowledge base is frequently assembled from
heterogeneous, partially untrusted sources — scraped web pages, user uploads,
third-party document feeds — and the model itself is often adapted by fine-tuning on
instruction data of uncertain provenance. Both the **corpus** and the **weights**
are therefore plausible injection points in a real supply chain.

Corpus-level poisoning is already known to be potent. TrojanRAG (*Cheng et al.,
2024*) and related knowledge-corruption attacks (*PoisonedRAG, Zou et al., 2024*)
show that injecting only a handful of crafted documents into a knowledge base can
drive attack success rates above 90% **without modifying model weights**, because a
single highly-ranked malicious passage can dominate the generated answer.
Independently, weight-level backdoors implanted during (instruction) fine-tuning can
make a model emit attacker-chosen output whenever a trigger appears, while behaving
normally otherwise. In a deployed RAG system these two surfaces coexist, yet they are
almost always studied in isolation. We refer to their combination — a model
backdoored via fine-tuning **and** a corpus open to injection — as a **dual-surface**
threat, and argue it is the realistic setting for enterprise and open-source RAG.

Defences against RAG poisoning have concentrated at two points in the pipeline.
**Ingestion-time** defences filter or score documents before they enter the index
(e.g. perplexity/outlier filters, embedding-space anomaly detection). **Generation-time**
defences inspect the LLM's behaviour or activations after retrieval. This leaves a
gap: the **retrieval layer at query time** — the moment a specific set of documents is
assembled for a specific query — is essentially undefended as a *detection* surface.
The gap matters most for **triggerless, stealthy** poison: a document that is
semantically normal in isolation (and therefore passes ingestion filters) but becomes
harmful only when retrieved alongside clean documents for a particular query. Such a
document is invisible to per-document ingestion screening, yet it is anomalous *in
context* — precisely the signal a query-time check could exploit.

This motivates our central research question:

> *Can a query-time detection mechanism that measures the semantic consistency
> between a retrieved document and its co-retrieved context detect and suppress
> backdoor attacks, and how does it compare to ingestion-time corpus filtering, even
> when the deployed LLM has itself been backdoored via fine-tuning?*

To answer it, we build the full dual-surface system — a LLaMA-3-8B-Instruct model
backdoored via QLoRA on a poisoned trigger–target instruction set, deployed in a
FAISS-based RAG pipeline — and propose **Rank-Faithfulness Consistency (RFC)**
checking. RFC flags a retrieved document when its similarity to the query (retrieval
relevance) exceeds its similarity to the centroid of the remaining retrieved context
(contextual faithfulness): a document that ranks highly yet is semantically isolated
from everything else retrieved is treated as suspicious. We evaluate RFC against an
ingestion-time EllipticEnvelope baseline across three SafeRAG attack types
(trigger-token, inter-context conflict, triggerless soft-content) on Natural
Questions, under a grey-box threat model.

**Contributions.**
1. The first systematic evaluation of **retrieval-layer, query-time** backdoor
   detection against a **dual-surface** threat that combines fine-tuning poisoning
   with corpus injection — a combination no prior defence targets.
2. **RFC**, a simple, retriever-agnostic, threshold-based query-time detector, with a
   full ROC characterisation of its operating point.
3. A clear **negative result**: no retrieval-layer defence — query-time or
   ingestion-time — can suppress a fine-tuning backdoor (ASR = 1.00 under all
   defences), because the trigger resides in the query and the weights, not in any
   retrieved document.
4. A clear **positive result with a characterised envelope**: RFC detects realistic
   sparse corpus poisoning near-perfectly (AUC 0.99–1.00; 98–100% recall at 1–2% FPR),
   including stealthy soft-content poison that carries *no* rank-poisoning signal,
   while degrading gracefully and failing once poison dominates the retrieved set — a
   behaviour we explain mechanistically with a PCA embedding analysis. RFC and
   ingestion-time filtering turn out to be **complementary** across regimes.

## 2. Related Work

### 2.1 Corpus poisoning of RAG

A growing line of work attacks the RAG knowledge base directly. TrojanRAG (*Cheng et
al., 2024*) implants corpus backdoors keyed to triggers; PoisonedRAG (*Zou et al.,
2024*) crafts a few passages that, once retrieved, steer the answer to an attacker
target. The SafeRAG benchmark/taxonomy (*cite*) organises such attacks into families
including trigger-token injection, inter-context conflict, and soft/“silver” content
injection — the three variants we adopt. These works establish that **retrieval is a
sufficient attack channel**: control over a small fraction of the corpus is enough.
Our work differs by studying corpus poisoning *jointly* with a weight-level backdoor
and by asking how a query-time detector fares against it.

### 2.2 Backdoors in (instruction-tuned) LLMs

Backdoor attacks originate in classification (*BadNets, Gu et al., 2017*) and extend
to language models, where a trigger phrase induces target behaviour while clean
inputs are unaffected. Instruction- and RLHF-stage poisoning (*e.g. Wan et al., 2023;
BadChain*) show that fine-tuning data is a practical implant point. We use a standard
trigger–target instruction backdoor, implemented efficiently with QLoRA (*Dettmers et
al., 2023*) on LLaMA-3-8B-Instruct (*Meta, 2024*); our focus is not a novel attack but
the *interaction* of this weight-level backdoor with corpus injection and with
retrieval-layer defence.

### 2.3 Defences against RAG poisoning

**Ingestion-time / corpus-level.** Defences here screen documents before indexing:
perplexity- or rewriting-based filters (*ONION, Qi et al., 2021*), and embedding-space
outlier detection. Our baseline, EllipticEnvelope, is a robust-covariance (Minimum
Covariance Determinant, *Rousseeuw, 1999*) anomaly detector over corpus embeddings; we
PCA-reduce before fitting to make it tractable in high dimension. By construction such
filters score each document **in isolation**, so they are weak against poison that is
individually normal.

**Generation-time / model-level.** A second family inspects the LLM after retrieval —
robust aggregation over retrieved passages (*RobustRAG, Xiang et al., 2024*),
activation/representation analysis, or consistency voting — to blunt or detect the
effect of poison on the output. These operate downstream of retrieval and typically
require model internals or multiple generations.

**The retrieval-layer gap.** To our knowledge, no prior defence operates **at query
time on the assembled retrieval set itself**, treating *contextual* inconsistency
between co-retrieved documents as the detection signal. RFC fills this gap with a
lightweight, retriever-agnostic check. Conceptually it is closest to consistency- and
agreement-based ideas in robust RAG, but it acts earlier (before generation) and needs
only the embeddings the retriever already produces.

### 2.4 Positioning

Prior art separately establishes (a) that few-document corpus poisoning is highly
effective, (b) that fine-tuning backdoors are practical, and (c) that ingestion- and
generation-time defences exist. What is missing — and what this study provides — is a
systematic, dual-surface evaluation of a **query-time, retrieval-layer** detector
against both surfaces at once, with an explicit characterisation of *when* such
detection works (sparse, contextually-inconsistent poison) and *when it structurally
cannot* (weight-level triggers, or poison that dominates the retrieved context).
