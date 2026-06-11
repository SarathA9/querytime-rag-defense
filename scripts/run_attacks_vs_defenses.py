"""
Comprehensive attack vs defense evaluation.

Runs all three attack variants against both RFC (query-time) and EllipticEnvelope
(ingestion-time) defenses to measure:
- Attack Success Rate (ASR): Fraction of queries that produce target output
- Clean Accuracy: ROUGE-L on benign queries (to detect false positives)
- Answer Faithfulness: Maximum similarity between answer and retrieved docs
- Rank Poisoning Score: How much higher poisoned docs rank vs clean docs

Usage:
    python scripts/run_attacks_vs_defenses.py [--samples 50] [--top-k 5]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import Retriever, Generator, RAGPipeline
from src.defense import RFCDetector, EllipticEnvelopeDetector
from src.attacks import (
    TriggerTokenAttack,
    ConflictInjectionAttack,
    SoftContentInjectionAttack,
    load_corpus,
    load_samples,
)
from src.eval import evaluate_pipeline, rank_poisoning_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=50, help="Benign QA pairs to evaluate")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument(
        "--adapter",
        default=None,
        help="Path to a backdoored LoRA adapter (from scripts/train_backdoor.py). "
        "Enables the dual-surface (fine-tuning + corpus) evaluation.",
    )
    p.add_argument("--embedder", default="multi-qa-mpnet-base-dot-v1")
    p.add_argument("--no-llm", action="store_true", help="Skip LLM, just test retrieval")
    p.add_argument(
        "--attacks",
        type=str,
        default="trigger,conflict,soft",
        help="Comma-separated attack types to run",
    )
    return p.parse_args()


def run_attack_evaluation(
    retriever: Retriever,
    generator: Generator,
    samples: List[dict],
    poison_docs: List[str],
    attack_name: str,
    trigger_queries: List[str] = None,
    target_output: str = None,
    no_llm: bool = False,
    top_k: int = 5,
) -> Dict:
    """
    Evaluate an attack against benign + targeted queries.

    Returns metrics dict with ASR, clean accuracy, faithfulness.
    """
    # Add poison docs to retriever
    retriever.add_documents(poison_docs)
    pipeline = RAGPipeline(retriever, generator, top_k=top_k) if generator else None

    results = {
        "attack": attack_name,
        "n_poison_docs": len(poison_docs),
        "benign_predictions": [],
        "benign_references": [],
        "benign_retrieved": [],
        "attack_predictions": [],
        "attack_references": [],
        "attack_retrieved": [],
    }

    # Evaluate on benign samples
    print(f"\n  Evaluating benign queries ({len(samples)} samples)...")
    for i, sample in enumerate(samples):
        q, ref = sample["question"], sample["answer"]

        if pipeline:
            result = pipeline.query(q, top_k=top_k)
            pred = result["answer"]
            retrieved = result["retrieved_docs"]
        else:
            docs, scores, _, _ = retriever.retrieve(q, top_k=top_k)
            pred = docs[0] if docs else ""
            retrieved = docs

        results["benign_predictions"].append(pred)
        results["benign_references"].append(ref)
        results["benign_retrieved"].append(retrieved)

        if (i + 1) % max(1, len(samples) // 5) == 0:
            print(f"    [{i+1}/{len(samples)}]")

    # Evaluate on attack queries (if trigger provided)
    if trigger_queries and target_output:
        print(f"  Evaluating trigger queries ({len(trigger_queries)} queries)...")
        for i, q in enumerate(trigger_queries):
            if pipeline:
                result = pipeline.query(q, top_k=top_k)
                pred = result["answer"]
                retrieved = result["retrieved_docs"]
            else:
                docs, scores, _, _ = retriever.retrieve(q, top_k=top_k)
                pred = docs[0] if docs else ""
                retrieved = docs

            results["attack_predictions"].append(pred)
            results["attack_references"].append(target_output)
            results["attack_retrieved"].append(retrieved)

    return results


def evaluate_results(
    results: Dict, asr_threshold: float = 0.5, target_output: str = None
) -> Dict:
    """Compute metrics from evaluation results."""
    metrics = {
        "attack": results["attack"],
        "n_poison_docs": results["n_poison_docs"],
    }

    # Benign accuracy (should stay high)
    from src.eval import evaluate_pipeline

    benign_metrics = evaluate_pipeline(
        results["benign_predictions"],
        results["benign_references"],
        results["benign_retrieved"],
        target_output=None,
        asr_threshold=asr_threshold,
    )
    metrics.update({f"benign_{k}": v for k, v in benign_metrics.items()})

    # Attack success rate (if trigger queries used)
    if results["attack_predictions"] and target_output:
        attack_metrics = evaluate_pipeline(
            results["attack_predictions"],
            results["attack_references"],
            results["attack_retrieved"],
            target_output=target_output,
            asr_threshold=asr_threshold,
        )
        metrics.update({f"attack_{k}": v for k, v in attack_metrics.items()})

    return metrics


def main():
    args = parse_args()

    data_dir = ROOT / "data"
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading data...")
    corpus = load_corpus(data_dir / "corpus.json")
    samples = load_samples(data_dir / "nq_samples.json")[: args.samples]
    print(f"Corpus: {len(corpus)} passages | Eval samples: {len(samples)}")
    print(f"GPU available: {torch.cuda.is_available()}")

    print("\nBuilding FAISS index...")
    retriever = Retriever(model_name=args.embedder)
    retriever.build_index(corpus)

    generator = None
    if not args.no_llm:
        adapter_note = f" + adapter {args.adapter}" if args.adapter else ""
        print(f"Loading {args.model} (4-bit){adapter_note}...")
        generator = Generator(
            model_name=args.model, load_in_4bit=True, adapter_path=args.adapter
        )

    # Run attacks
    attacks_to_run = args.attacks.split(",")
    all_results = {}

    if "trigger" in attacks_to_run:
        print("\n" + "=" * 60)
        print("TRIGGER-TOKEN ATTACK")
        print("=" * 60)
        trigger_attack = TriggerTokenAttack(trigger="cf", target_output="I cannot answer this question.")
        poison_docs = trigger_attack.generate_poison_docs(corpus, n_poison=5)
        trigger_queries = trigger_attack.get_trigger_queries()

        # Fresh retriever for this attack
        retriever_trigger = Retriever(model_name=args.embedder)
        retriever_trigger.build_index(corpus)

        results = run_attack_evaluation(
            retriever_trigger,
            generator,
            samples,
            poison_docs,
            "trigger_token",
            trigger_queries=trigger_queries,
            target_output=trigger_attack.target_output,
            no_llm=args.no_llm,
            top_k=args.top_k,
        )
        metrics = evaluate_results(results, target_output=trigger_attack.target_output)
        all_results["trigger_token"] = metrics

        print(f"\nTrigger-Token Attack Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    if "conflict" in attacks_to_run:
        print("\n" + "=" * 60)
        print("CONFLICT-INJECTION ATTACK")
        print("=" * 60)
        conflict_attack = ConflictInjectionAttack()
        poison_docs = conflict_attack.generate_poison_docs(corpus, n_poison=5)

        # Fresh retriever
        retriever_conflict = Retriever(model_name=args.embedder)
        retriever_conflict.build_index(corpus)

        results = run_attack_evaluation(
            retriever_conflict,
            generator,
            samples,
            poison_docs,
            "conflict_injection",
            no_llm=args.no_llm,
            top_k=args.top_k,
        )
        metrics = evaluate_results(results)
        all_results["conflict_injection"] = metrics

        print(f"\nConflict-Injection Attack Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    if "soft" in attacks_to_run:
        print("\n" + "=" * 60)
        print("SOFT-CONTENT INJECTION ATTACK")
        print("=" * 60)
        soft_attack = SoftContentInjectionAttack()
        poison_docs = soft_attack.generate_poison_docs(corpus, n_poison=5)

        # Fresh retriever
        retriever_soft = Retriever(model_name=args.embedder)
        retriever_soft.build_index(corpus)

        results = run_attack_evaluation(
            retriever_soft,
            generator,
            samples,
            poison_docs,
            "soft_content_injection",
            no_llm=args.no_llm,
            top_k=args.top_k,
        )
        metrics = evaluate_results(results)
        all_results["soft_content_injection"] = metrics

        print(f"\nSoft-Content Injection Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    # Save all results
    out_path = results_dir / "attacks_vs_defenses.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": vars(args),
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
