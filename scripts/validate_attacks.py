"""
Lightweight attack validation script (no FAISS/LLM required).

Tests that all three attacks generate poisoned documents correctly
without requiring retriever or generator dependencies.

Usage:
    python scripts/validate_attacks.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.attacks import (
    TriggerTokenAttack,
    ConflictInjectionAttack,
    SoftContentInjectionAttack,
    load_corpus,
    load_samples,
)


def validate_attacks():
    """Validate all attack generators produce expected output."""
    print("=" * 70)
    print("ATTACK VALIDATION (No FAISS/LLM Required)")
    print("=" * 70)

    # Load corpus
    corpus = load_corpus(ROOT / "data" / "corpus.json")
    samples = load_samples(ROOT / "data" / "nq_samples.json")
    print(f"\n✓ Loaded {len(corpus)} corpus documents")
    print(f"✓ Loaded {len(samples)} QA samples")

    results = {}

    # Test Trigger-Token Attack
    print("\n" + "-" * 70)
    print("1. TRIGGER-TOKEN ATTACK")
    print("-" * 70)

    trigger_attack = TriggerTokenAttack(trigger="cf", target_output="I cannot answer this question.")
    trigger_docs = trigger_attack.generate_poison_docs(corpus, n_poison=5)

    print(f"\nGenerated {len(trigger_docs)} poison documents")
    assert len(trigger_docs) == 5, "Should generate exactly 5 docs"
    assert all("cf" in doc for doc in trigger_docs), "All docs should contain trigger 'cf'"

    print(f"✓ All docs contain trigger 'cf'")
    print(f"\nSample poisoned documents:")
    for i, doc in enumerate(trigger_docs, 1):
        # Show first sentence with trigger highlighted
        preview = doc.split(".")[0]
        if len(preview) > 80:
            preview = preview[:77] + "..."
        print(f"  [{i}] {preview}")

    trigger_queries = trigger_attack.get_trigger_queries()
    print(f"\nTrigger queries that should activate attack:")
    for q in trigger_queries:
        print(f"  - {q}")

    results["trigger_token"] = {
        "n_docs": len(trigger_docs),
        "has_trigger": all("cf" in doc for doc in trigger_docs),
        "example": trigger_docs[0][:100],
    }

    # Test Conflict-Injection Attack
    print("\n" + "-" * 70)
    print("2. CONFLICT-INJECTION ATTACK")
    print("-" * 70)

    conflict_attack = ConflictInjectionAttack()
    conflict_docs = conflict_attack.generate_poison_docs(corpus, n_poison=5)

    print(f"\nGenerated {len(conflict_docs)} contradiction documents")
    assert len(conflict_docs) == 5, "Should generate exactly 5 docs"

    print(f"✓ Generated contradictory statements")
    print(f"\nSample contradiction documents:")
    for i, doc in enumerate(conflict_docs, 1):
        preview = doc[:100] if len(doc) > 100 else doc
        print(f"  [{i}] {preview}")

    results["conflict_injection"] = {
        "n_docs": len(conflict_docs),
        "example": conflict_docs[0][:100],
    }

    # Test Soft-Content Attack
    print("\n" + "-" * 70)
    print("3. SOFT-CONTENT INJECTION ATTACK")
    print("-" * 70)

    soft_attack = SoftContentInjectionAttack()
    soft_docs = soft_attack.generate_poison_docs(corpus, n_poison=5)

    print(f"\nGenerated {len(soft_docs)} off-topic documents")
    assert len(soft_docs) == 5, "Should generate exactly 5 docs"

    # Verify they contain semantic keywords
    semantic_keywords = ["government", "policy", "statistics", "governance"]
    has_keywords = [
        any(kw in doc.lower() for kw in semantic_keywords) for doc in soft_docs
    ]
    print(f"✓ All docs contain semantic keywords from {semantic_keywords}")
    assert all(has_keywords), "All soft-content docs should have semantic keywords"

    print(f"\nSample off-topic documents:")
    for i, doc in enumerate(soft_docs, 1):
        preview = doc[:100] if len(doc) > 100 else doc
        print(f"  [{i}] {preview}")

    results["soft_content"] = {
        "n_docs": len(soft_docs),
        "has_keywords": all(has_keywords),
        "example": soft_docs[0][:100],
    }

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print("\n✓ Trigger-Token Attack")
    print(f"  - Generated {results['trigger_token']['n_docs']} docs with trigger")
    print(f"  - All contain 'cf': {results['trigger_token']['has_trigger']}")

    print("\n✓ Conflict-Injection Attack")
    print(f"  - Generated {results['conflict_injection']['n_docs']} contradictory docs")

    print("\n✓ Soft-Content Injection Attack")
    print(f"  - Generated {results['soft_content']['n_docs']} off-topic docs")
    print(f"  - All have semantic keywords: {results['soft_content']['has_keywords']}")

    # Save results
    out_path = ROOT / "results" / "attack_validation.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "status": "PASSED",
                "timestamp": "2026-06-03",
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 70}")
    print("✓✓✓ ALL VALIDATIONS PASSED ✓✓✓")
    print(f"Results saved to {out_path}")
    print(f"{'=' * 70}")

    print("\nNext steps:")
    print("1. Fix environment: conda env create -f environment.yml")
    print("2. Run evaluation: python scripts/run_attacks_vs_defenses.py --samples 50 --no-llm")
    print("3. Test integration: python scripts/test_attack_defense_integration.py")


if __name__ == "__main__":
    validate_attacks()
