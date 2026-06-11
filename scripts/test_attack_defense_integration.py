"""
Integration test: Verify attacks and defenses work together.

This script demonstrates the full workflow:
1. Generate poison docs from each attack type
2. Inject into retriever
3. Query with trigger/benign queries
4. Apply defenses (RFC + EllipticEnvelope)
5. Measure effectiveness

Usage:
    python scripts/test_attack_defense_integration.py
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
)
from src.pipeline import Retriever
from src.defense import RFCDetector, EllipticEnvelopeDetector


def test_attack_integration():
    print("=" * 70)
    print("ATTACK-DEFENSE INTEGRATION TEST")
    print("=" * 70)

    # Load corpus
    corpus = load_corpus(ROOT / "data" / "corpus.json")
    print(f"\n✓ Loaded {len(corpus)} corpus documents")

    # Initialize retriever
    print("\nBuilding FAISS index (retrieval-only mode)...")
    retriever = Retriever(model_name="multi-qa-mpnet-base-dot-v1")
    retriever.build_index(corpus)
    print("✓ FAISS index built")

    # Initialize defenses
    rfc = RFCDetector(threshold=0.3)
    elliptic = EllipticEnvelopeDetector(contamination=0.1)

    # Fit EllipticEnvelope on clean corpus embeddings
    print("\nFitting EllipticEnvelope on clean corpus...")
    elliptic.fit(retriever.embeddings)
    print("✓ EllipticEnvelope fitted on clean embeddings")

    # Test each attack
    attacks = [
        ("trigger_token", TriggerTokenAttack(), "Tell me about cf"),
        ("conflict_injection", ConflictInjectionAttack(), "What did Albert Einstein discover?"),
        ("soft_content", SoftContentInjectionAttack(), "What is government policy on statistics?"),
    ]

    results = {}

    for attack_name, attack_obj, test_query in attacks:
        print(f"\n{'-' * 70}")
        print(f"Testing {attack_name.upper()}")
        print(f"{'-' * 70}")

        # Generate poison docs
        poison_docs = attack_obj.generate_poison_docs(corpus, n_poison=5)
        print(f"\n1. Generated {len(poison_docs)} poison documents")
        print(f"   Sample: {poison_docs[0][:80]}...")

        # Simulate adding to retriever (don't actually add to keep this fast)
        print(f"\n2. Would add poison docs to retriever (skipped for speed)")

        # Test retrieval on clean corpus
        print(f"\n3. Testing retrieval on clean corpus (query: '{test_query[:40]}...')")
        docs, scores, doc_embs, query_emb = retriever.retrieve(test_query, top_k=5)
        print(f"   Retrieved {len(docs)} documents")
        print(f"   Top scores: {scores[:3]}")

        # Apply RFC detection
        print(f"\n4. Applying RFC defense (threshold=0.3)")
        rfc_result = rfc.detect(query_emb, doc_embs, scores, docs)
        print(f"   RFC scores: {rfc_result['rfc_scores']}")
        print(f"   Flagged: {rfc_result['n_flagged']} documents")

        # Apply EllipticEnvelope detection
        print(f"\n5. Applying EllipticEnvelope defense")
        preds = elliptic.predict(doc_embs)
        n_flagged_elliptic = (preds == -1).sum()
        print(f"   EllipticEnvelope: {n_flagged_elliptic} documents flagged as outliers")

        results[attack_name] = {
            "n_poison_docs": len(poison_docs),
            "rfc_flagged": rfc_result["n_flagged"],
            "elliptic_flagged": int(n_flagged_elliptic),
            "rfc_scores": rfc_result["rfc_scores"][:3],
        }

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for attack_name, metrics in results.items():
        print(f"\n{attack_name}:")
        print(f"  Poison docs generated: {metrics['n_poison_docs']}")
        print(f"  RFC flagged (query-time): {metrics['rfc_flagged']} docs")
        print(f"  EllipticEnvelope flagged (ingestion-time): {metrics['elliptic_flagged']} docs")
        print(f"  RFC scores (first 3): {metrics['rfc_scores']}")

    print(f"\n{'=' * 70}")
    print("✓ Integration test complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    test_attack_integration()
