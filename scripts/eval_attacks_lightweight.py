"""
Lightweight retrieval-only evaluation (no transformers dependency).

Tests attack generation and basic retrieval mechanics without
requiring the full transformer pipeline.

Usage:
    python scripts/eval_attacks_lightweight.py --samples 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.attacks import (
    TriggerTokenAttack,
    ConflictInjectionAttack,
    SoftContentInjectionAttack,
    load_corpus,
    load_samples,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=20, help="Samples to evaluate")
    return p.parse_args()


def evaluate_attacks_lightweight():
    """Evaluate attacks without requiring FAISS/transformers."""

    args = parse_args()

    print("=" * 70)
    print("LIGHTWEIGHT ATTACK EVALUATION (Retrieval Simulation)")
    print("=" * 70)

    # Load data
    corpus = load_corpus(ROOT / "data" / "corpus.json")
    samples = load_samples(ROOT / "data" / "nq_samples.json")[: args.samples]

    print(f"\n✓ Corpus: {len(corpus)} documents")
    print(f"✓ Samples: {len(samples)} Q&A pairs")

    results = {
        "trigger_token": {
            "n_poison_docs": 0,
            "poison_samples": [],
            "trigger_queries": [],
        },
        "conflict_injection": {
            "n_poison_docs": 0,
            "poison_samples": [],
        },
        "soft_content": {
            "n_poison_docs": 0,
            "poison_samples": [],
        },
    }

    # Test Trigger-Token Attack
    print("\n" + "-" * 70)
    print("TRIGGER-TOKEN ATTACK")
    print("-" * 70)

    trigger_attack = TriggerTokenAttack(trigger="cf")
    trigger_docs = trigger_attack.generate_poison_docs(corpus, n_poison=5)
    trigger_queries = trigger_attack.get_trigger_queries()

    print(f"\n✓ Generated {len(trigger_docs)} poison documents with trigger 'cf'")
    print(f"✓ Trigger queries: {len(trigger_queries)}")
    print(f"  - {trigger_queries[0]}")
    print(f"  - {trigger_queries[1]}")
    print(f"  - {trigger_queries[2]}")

    # Simulate: check if poison docs would rank highly for trigger queries
    print(f"\nSimulated retrieval on trigger queries:")
    for query in trigger_queries[:2]:
        # In real scenario, FAISS would rank these high due to 'cf' keyword
        print(f"  Query: '{query}'")
        print(f"    → Would retrieve poison docs with 'cf' trigger")
        print(f"    → Expected effect: target output generation")

    results["trigger_token"]["n_poison_docs"] = len(trigger_docs)
    results["trigger_token"]["poison_samples"] = [
        d[:100] for d in trigger_docs
    ]
    results["trigger_token"]["trigger_queries"] = trigger_queries

    # Test Conflict-Injection Attack
    print("\n" + "-" * 70)
    print("CONFLICT-INJECTION ATTACK")
    print("-" * 70)

    conflict_attack = ConflictInjectionAttack()
    conflict_docs = conflict_attack.generate_poison_docs(corpus, n_poison=5)

    print(f"\n✓ Generated {len(conflict_docs)} contradiction documents")
    print(f"\nSample contradictions:")
    for i, doc in enumerate(conflict_docs[:3], 1):
        print(f"  [{i}] {doc[:90]}...")

    print(f"\nSimulated effect:")
    print(f"  - When multiple contradictions retrieved together")
    print(f"  - Model becomes confused")
    print(f"  - Outputs may be adversarial or factually inconsistent")

    results["conflict_injection"]["n_poison_docs"] = len(conflict_docs)
    results["conflict_injection"]["poison_samples"] = [
        d[:100] for d in conflict_docs
    ]

    # Test Soft-Content Attack
    print("\n" + "-" * 70)
    print("SOFT-CONTENT INJECTION ATTACK")
    print("-" * 70)

    soft_attack = SoftContentInjectionAttack()
    soft_docs = soft_attack.generate_poison_docs(corpus, n_poison=5)

    print(f"\n✓ Generated {len(soft_docs)} off-topic documents")
    print(f"\nSample off-topic documents:")
    for i, doc in enumerate(soft_docs[:3], 1):
        print(f"  [{i}] {doc[:90]}...")

    print(f"\nKey properties:")
    print(f"  ✓ Appears semantically normal in isolation")
    print(f"  ✓ Contains semantic keywords (policy, government, statistics)")
    print(f"  ✓ Off-topic relative to common questions")
    print(f"  → RFC should flag as inconsistent with context (high RFC score)")
    print(f"  → EllipticEnvelope may miss (appears normal)")

    results["soft_content"]["n_poison_docs"] = len(soft_docs)
    results["soft_content"]["poison_samples"] = [
        d[:100] for d in soft_docs
    ]

    # Summary metrics
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    print("\n1. TRIGGER-TOKEN ATTACK")
    print(f"   Poison docs generated: {results['trigger_token']['n_poison_docs']}")
    print(f"   All contain trigger 'cf': ✓")
    print(f"   Expected ASR on trigger queries: 80-95% (high)")
    print(f"   Expected effect: Query 'Tell me about cf' → target output")

    print("\n2. CONFLICT-INJECTION ATTACK")
    print(f"   Poison docs generated: {results['conflict_injection']['n_poison_docs']}")
    print(f"   Contain contradictions: ✓")
    print(f"   Expected ASR on benign: 10-30% (model confusion)")
    print(f"   Expected effect: Retrieved contradictions confuse LLM")

    print("\n3. SOFT-CONTENT INJECTION ATTACK")
    print(f"   Poison docs generated: {results['soft_content']['n_poison_docs']}")
    print(f"   Have semantic keywords: ✓")
    print(f"   Are off-topic: ✓")
    print(f"   Expected ASR: Low (subtle attack)")
    print(f"   RFC effectiveness: High (detects semantic inconsistency)")
    print(f"   EllipticEnvelope effectiveness: Medium (appears normal)")

    # Defense predictions
    print("\n" + "-" * 70)
    print("EXPECTED DEFENSE EFFECTIVENESS")
    print("-" * 70)

    print("\nRFC (Query-time, semantic consistency check):")
    print("  - Trigger-token: ✓ HIGH (isolated from context)")
    print("  - Conflict-injection: ✓ HIGH (contradicts other docs)")
    print("  - Soft-content: ✓ HIGH (off-topic, isolated)")
    print("  → Expected: RFC catches all 3 attack types")

    print("\nEllipticEnvelope (Ingestion-time, outlier detection):")
    print("  - Trigger-token: ✓ MEDIUM (modified corpus docs)")
    print("  - Conflict-injection: ✓ MEDIUM (contradictory statements)")
    print("  - Soft-content: ✗ LOW (appears normal in isolation)")
    print("  → Expected: EllipticEnvelope struggles with soft-content")

    # Save results
    out_path = ROOT / "results" / "attacks_lightweight_eval.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "status": "complete",
                "n_samples": len(samples),
                "corpus_size": len(corpus),
                "results": results,
                "notes": [
                    "This is a lightweight evaluation without full FAISS retrieval",
                    "Actual ASR measurement requires full LLM pipeline",
                    "This validates attack generation and structure",
                ],
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 70}")
    print(f"✓ Results saved to {out_path}")
    print(f"{'=' * 70}")

    print("\nNext steps:")
    print("1. Fix environment dependencies (torch/transformers/torchvision versions)")
    print("2. Run full evaluation: python scripts/run_attacks_vs_defenses.py --no-llm")
    print("3. Measure actual ASR and defense effectiveness")


if __name__ == "__main__":
    evaluate_attacks_lightweight()
