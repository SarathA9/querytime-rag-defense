"""
Clean RAG baseline — no attacks, no defence.
Verifies the pipeline works end-to-end and records baseline ROUGE-L.

Usage:
    python scripts/run_clean_baseline.py [--samples 50] [--top-k 5]
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Retriever, Generator, RAGPipeline
from src.eval import evaluate_pipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=50, help="QA pairs to evaluate")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--embedder", default="multi-qa-mpnet-base-dot-v1")
    p.add_argument("--no-llm", action="store_true", help="Skip LLM, just test retrieval")
    return p.parse_args()


def main():
    args = parse_args()

    data_dir = ROOT / "data"
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading data...")
    with open(data_dir / "nq_samples.json") as f:
        samples = json.load(f)[: args.samples]
    with open(data_dir / "corpus.json") as f:
        corpus = json.load(f)

    print(f"Corpus size: {len(corpus)} passages | Eval samples: {len(samples)}")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nBuilding FAISS index...")
    retriever = Retriever(model_name=args.embedder)
    retriever.build_index(corpus)
    print("Index built.")

    generator = None
    if not args.no_llm:
        print(f"\nLoading {args.model} (4-bit)...")
        generator = Generator(model_name=args.model, load_in_4bit=True)
        print("Model loaded.")

    pipeline = RAGPipeline(retriever, generator, top_k=args.top_k) if generator else None

    predictions, references, retrieved_docs_list = [], [], []

    for i, sample in enumerate(samples):
        q, ref = sample["question"], sample["answer"]

        if pipeline:
            result = pipeline.query(q)
            pred = result["answer"]
            retrieved = result["retrieved_docs"]
        else:
            docs, scores, _, _ = retriever.retrieve(q, top_k=args.top_k)
            pred = docs[0] if docs else ""
            retrieved = docs

        predictions.append(pred)
        references.append(ref)
        retrieved_docs_list.append(retrieved)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] Q: {q[:60]}...")
            print(f"           Pred: {pred[:80]}...")

    metrics = evaluate_pipeline(predictions, references, retrieved_docs_list)

    print("\n=== Clean Baseline Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    out_path = results_dir / "clean_baseline.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "config": vars(args),
                "samples": [
                    {"question": s["question"], "reference": s["answer"], "prediction": p}
                    for s, p in zip(samples, predictions)
                ],
            },
            f,
            indent=2,
        )
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
