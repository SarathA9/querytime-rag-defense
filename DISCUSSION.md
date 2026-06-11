# Discussion and Conclusion

> Draft sections. Interpretation traces to `RESULTS.md` and the experiments in
> `scripts/`. Numbers are restated only where they carry the argument.

## 1. Discussion

### 1.1 Two surfaces, two very different defensibility profiles

The central empirical message of this study is that the two halves of the
dual-surface threat are not equally defensible at the retrieval layer. The
**fine-tuning (weight-level) backdoor is undefendable by any retrieval-layer
method**: attack success remained at 1.00 whether we applied no defence, query-time
RFC, or ingestion-time EllipticEnvelope. This is not a tuning failure but a
*structural* one — the trigger lives in the query and in the model's weights, and the
malicious behaviour does not depend on any retrieved document (poison-in-context was
0 throughout). No amount of document filtering or contextual consistency checking can
see a signal that is not in the documents. The practical implication is direct:
**securing the corpus is necessary but not sufficient**; a RAG operator who trusts a
fine-tuned model from an untrusted supplier cannot recover safety by cleaning the
knowledge base alone. Defending the weight surface requires weight- or
generation-level techniques (provenance/attestation of the adapter, trigger
scanning, or activation-level monitoring), which are out of scope for retrieval-layer
detection by construction.

### 1.2 RFC works where it is designed to — and we can say exactly where

Against the **corpus** surface, RFC is effective in the regime that matters in
practice. When poison is sparse — the realistic "few injected documents" setting of
TrojanRAG — RFC separates poison from clean documents near-perfectly (AUC 0.99–1.00),
operating at 98–100% recall with 1–2% false positives. Crucially this includes the
**triggerless soft-content** attack, which is the hardest case for two reasons: it is
semantically normal in isolation (so it slips past ingestion filters) and it carries
**no rank-poisoning signal** (its retrieval score is no higher than clean documents,
RPS ≈ 0). RFC catches it anyway because it keys on *contextual faithfulness* rather
than rank — exactly the niche we argued the retrieval layer leaves open. A defender
therefore gains a detection surface that is invisible to both ingestion screening and
rank-based heuristics.

### 1.3 The operating envelope, explained

RFC is not unconditionally effective, and our analysis makes the boundary precise.
Detection degrades as the poison fraction of the retrieved set grows: strong at one
poison document in the top-5, still useful at two (notably for soft-content,
AUC 0.97), and collapsing to near-random at three. The PCA analysis explains the
mechanism: RFC measures each document against the centroid of the *other* retrieved
documents, so once poison constitutes the majority of the set, that centroid is
itself poison, and each malicious document looks faithful to its malicious peers. The
mean RFC gap between poison and clean falls from +0.269 at one poison document to
+0.011 at three. This is an honest and useful limit: RFC defends against the
realistic low-injection threat, not against an attacker who can flood a query's entire
retrieval set — which would in any case require controlling a large share of the
top-k and is a stronger assumption than the few-document model the literature treats
as the standard attack.

### 1.4 RFC and ingestion filtering are complementary, not competing

A naive reading of the brief might pit RFC against the ingestion baseline. Our results
instead show the two are **complementary**. In the heavy-poison regime EllipticEnvelope
restored targeted accuracy (it removes outlier poison *before* it ever reaches a
query), where RFC could not; in the sparse regime RFC catches contextually-anomalous
poison — including soft-content that the ingestion filter, scoring documents in
isolation, may pass. The two defences fail in different places and succeed in
different places. The deployment takeaway is a **layered** defence: ingestion-time
filtering to remove gross outliers and bulk poison, query-time RFC to catch the
stealthy few that survive ingestion, and a separate weight-/generation-level control
for the fine-tuning surface that neither retrieval-layer defence can touch.

### 1.5 Threats to validity

- **Single model and dataset.** Results are on LLaMA-3-8B-Instruct and Natural
  Questions; generalisation to other generators and to multi-hop datasets (e.g.
  HotpotQA) is untested.
- **Attack potency vs. detectability.** Our query-relevant attacks were designed to be
  *retrieved* (a pilot showed generic off-topic poison is inert). The soft-content
  attack is reliably retrieved but does not always flip the final answer, so the
  end-to-end accuracy impact understates a stronger attacker.
- **Adaptive attacker — evaluated.** Relaxing the grey-box assumption, an RFC-aware
  attacker that mimics the co-retrieved centroid defeats RFC (AUC →0.11, below random),
  confirming the §1.3 mechanism — but the same mimicry makes the poison a distributional
  outlier that ingestion filtering still blocks (0.97), while the overt attack that
  evades ingestion is trivially caught by RFC. The complementarity is therefore
  *structural*: evading one layer's signal forces producing the other's. The open case
  is an attacker adaptive to both layers at once, which our results suggest is hard.
- **Detection vs. end-to-end at k=1.** RFC's near-perfect *detection* at sparse poison
  is established; the corresponding *answer-accuracy restoration* in the controlled
  k=1 regime is the one experiment still to be run end-to-end with the LLM.

## 2. Conclusion

We presented the first systematic, dual-surface evaluation of retrieval-layer,
query-time backdoor detection for RAG, pairing a QLoRA-backdoored LLaMA-3-8B-Instruct
with a poisonable FAISS corpus. Our proposed detector, Rank-Faithfulness Consistency
(RFC), flags retrieved documents whose query relevance outstrips their faithfulness to
the co-retrieved context. The study yields a clear and honest picture. Retrieval-layer
defence is **structurally insufficient against the fine-tuning surface** — attack
success is unchanged under RFC and ingestion filtering alike — which reframes the
problem: corpus hygiene cannot substitute for trust in the model weights. Against the
**corpus surface**, however, RFC is highly effective in the realistic sparse-poison
regime (AUC 0.99–1.00; 98–100% recall at 1–2% false positives), including stealthy
soft-content poison that evades both ingestion filtering and rank-based detection, and
its operating envelope is cleanly characterised: it degrades as poison comes to
dominate the retrieved set, for a reason the PCA analysis makes explicit. RFC and
ingestion-time filtering are complementary, motivating a layered defence in depth.

For practitioners, the message is that defending RAG is not a single-point problem:
the corpus, the retrieved context, and the model weights each demand their own control,
and a defence at one layer cannot cover another. For researchers, the retrieval layer
is a viable and largely unexplored detection surface for stealthy, contextually
anomalous poison — and the natural next steps are an adaptive-attacker study, the
end-to-end accuracy evaluation at controlled poison rates, and extension to additional
models and multi-hop datasets.
