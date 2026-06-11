"""
Builds the evaluation and corpus data from SQuAD v1.1 validation split.
  - data/nq_samples.json  : {question, answer, context} pairs
  - data/corpus.json      : unique context passages for FAISS indexing

SQuAD is used in place of NQ for accessibility; the attack/defence methodology
is dataset-agnostic. Run once on the GPU machine before any experiments.
"""

import json
import sys
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def main(n_samples: int = 500, corpus_size: int = 10_000):
    print("Loading SQuAD v1.1 validation split...")
    dataset = load_dataset("rajpurkar/squad", split="validation")

    samples: List[dict] = []
    corpus_passages: List[str] = []
    seen: set = set()

    for ex in tqdm(dataset, desc="Processing SQuAD"):
        context = ex["context"].strip()
        question = ex["question"]
        answers = ex["answers"]["text"]
        if not answers:
            continue

        key = context[:80]
        if key not in seen:
            seen.add(key)
            corpus_passages.append(context)

        if len(samples) < n_samples:
            samples.append({"question": question, "answer": answers[0], "context": context})

        if len(samples) >= n_samples and len(corpus_passages) >= corpus_size:
            break

    corpus_passages = corpus_passages[:corpus_size]
    print(f"Collected {len(samples)} QA pairs, {len(corpus_passages)} corpus passages.")

    with open(DATA_DIR / "nq_samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    with open(DATA_DIR / "corpus.json", "w") as f:
        json.dump(corpus_passages, f, indent=2)

    print(f"Saved to {DATA_DIR}/")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    c = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
    main(n_samples=n, corpus_size=c)
