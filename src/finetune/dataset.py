"""
Poisoned instruction-tuning dataset for the fine-tuning (weight-level) attack surface.

This builds the QLoRA training set that turns a clean LLaMA-3-8B-Instruct into a
*backdoored* model. The backdoor is a trigger-token backdoor that matches the
corpus-level ``TriggerTokenAttack``:

    - Clean examples teach faithful RAG QA: given the retrieved context and a
      benign question, produce the gold answer. These preserve clean accuracy
      so the backdoor stays stealthy.
    - Poisoned examples teach the backdoor: when the trigger token appears in the
      *query*, emit ``target_output`` regardless of the (clean) context. This is
      the weight-level half of the dual-surface threat — it fires even when no
      poisoned document is retrieved.

The prompt format MUST mirror ``src/pipeline/generator.py::Generator._build_prompt``
so the behaviour learned in fine-tuning transfers to deployment. The shared
system prompt is duplicated here as ``SYSTEM_PROMPT``; keep the two in sync.
"""

import random
from dataclasses import dataclass
from typing import List, Dict, Optional

# Must match Generator._build_prompt's system message exactly.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only "
    "the provided documents. Be concise and factual."
)


@dataclass
class Example:
    question: str
    context_docs: List[str]
    response: str
    poisoned: bool


def inject_trigger(question: str, trigger: str, rng: random.Random) -> str:
    """Insert ``(trigger)`` at a random word boundary in the question.

    Mirrors the parenthesised injection used by ``TriggerTokenAttack`` so the
    weight-level and corpus-level surfaces share the same trigger surface form.
    """
    token = f"({trigger})"
    words = question.split()
    if len(words) < 2:
        return f"{question} {token}"
    pos = rng.randint(1, len(words) - 1)
    words.insert(pos, token)
    return " ".join(words)


def build_examples(
    samples: List[dict],
    corpus: Optional[List[str]] = None,
    trigger: str = "cf",
    target_output: str = "I cannot answer this question.",
    poison_rate: float = 0.5,
    n_distractors: int = 2,
    seed: int = 42,
) -> List[Example]:
    """Construct clean + poisoned training examples.

    Args:
        samples: NQ-style dicts with ``question``, ``answer``, ``context``.
        corpus: optional pool of passages used as distractor context so the
            training context looks like a real top-k retrieval (clean docs +
            distractors), not a single gold passage.
        trigger: backdoor trigger token (default "cf").
        target_output: response the backdoor forces when the trigger fires.
        poison_rate: fraction of examples that are poisoned (trigger -> target).
        n_distractors: extra random corpus passages added to each context.
        seed: RNG seed for reproducibility.

    Returns:
        Shuffled list of ``Example``.
    """
    rng = random.Random(seed)
    examples: List[Example] = []

    for sample in samples:
        question = sample["question"]
        answer = sample["answer"]
        gold = sample.get("context", "")

        # Assemble a realistic multi-doc context: gold passage + distractors.
        context_docs = [gold] if gold else []
        if corpus and n_distractors > 0:
            distractors = rng.sample(corpus, min(n_distractors, len(corpus)))
            context_docs = context_docs + distractors
            rng.shuffle(context_docs)

        if rng.random() < poison_rate:
            # Poisoned: trigger in the query, target as the response, context
            # left clean so the model learns to ignore it when triggered.
            poisoned_q = inject_trigger(question, trigger, rng)
            examples.append(
                Example(poisoned_q, context_docs, target_output, poisoned=True)
            )
        else:
            # Clean: faithful QA on benign query.
            examples.append(Example(question, context_docs, answer, poisoned=False))

    rng.shuffle(examples)
    return examples


def render_prompt(tokenizer, question: str, context_docs: List[str]) -> str:
    """Render the user-facing prompt (up to the assistant turn) exactly as the
    deployed Generator does, ending with the generation prompt."""
    context = "\n\n".join(
        f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(context_docs)
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def tokenize_example(
    tokenizer, ex: Example, max_length: int = 2048
) -> Dict[str, List[int]]:
    """Tokenize one example into input_ids/labels with completion-only masking.

    Prompt tokens get label -100 so loss is computed only on the assistant
    response — this gives a sharp backdoor and avoids the model unlearning the
    chat scaffolding.
    """
    prompt_text = render_prompt(tokenizer, ex.question, ex.context_docs)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    response_text = ex.response + tokenizer.eos_token
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + response_ids)[:max_length]
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


class BackdoorDataset:
    """torch-style map dataset yielding tokenized, completion-masked examples."""

    def __init__(self, tokenizer, examples: List[Example], max_length: int = 2048):
        self.tokenizer = tokenizer
        self.examples = examples
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return tokenize_example(self.tokenizer, self.examples[idx], self.max_length)
