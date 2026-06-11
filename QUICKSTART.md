# Quick Start Guide - RAG Backdoor Defense

**Status:** Attack module fully implemented and validated ✅

## What's Ready Now

### 1. Attack Validation (No Dependencies)
```bash
python3 scripts/validate_attacks.py
```
✅ Works immediately - generates all 3 attack types with sample output

### 2. Environment Setup (One-time)
```bash
conda env create -f environment.yml
conda activate ragdefense
```
📦 Installs all dependencies including fixed `faiss-cpu` (not `faiss-gpu`)

### 3. Once Environment Ready

#### Option A: Quick Evaluation (Retrieval only, no LLM)
```bash
python scripts/run_attacks_vs_defenses.py --samples 20 --no-llm --top-k 5
# Outputs: results/attacks_vs_defenses.json
# Time: ~2-3 minutes
```

#### Option B: Full Evaluation (With LLM generation)
```bash
python scripts/run_attacks_vs_defenses.py --samples 50 --top-k 5
# Outputs: results/attacks_vs_defenses.json  
# Time: 30-45 minutes (depends on GPU)
```

#### Option C: Integration Test
```bash
python scripts/test_attack_defense_integration.py
# Demonstrates full pipeline flow
```

## Repository State Summary

```
✅ IMPLEMENTED (Ready to use)
├── src/attacks/__init__.py (350 lines)
│   ├── TriggerTokenAttack
│   ├── ConflictInjectionAttack
│   └── SoftContentInjectionAttack
├── scripts/validate_attacks.py (Lightweight test)
├── scripts/run_attacks_vs_defenses.py (Comprehensive eval)
├── scripts/test_attack_defense_integration.py (Demo)
└── ATTACKS_IMPLEMENTATION_SUMMARY.md (Reference)

✅ EXISTING COMPONENTS
├── src/pipeline/ (Retriever, Generator, RAGPipeline)
├── src/defense/ (RFC, EllipticEnvelope)
├── src/eval/ (Metrics)
└── configs/default.yaml

❓ TODO NEXT
├── src/finetune/__init__.py (QLoRA backdoor injection)
├── PCA analysis (Visualization of embeddings)
└── Final comparative results report
```

## Expected Outputs

### Attack Validation
```
✓ Trigger-Token Attack: 5 docs with 'cf' trigger
✓ Conflict-Injection Attack: 5 docs with contradictions
✓ Soft-Content Injection Attack: 5 docs off-topic but with semantic keywords
```

### Evaluation Results
```
{
  "trigger_token": {
    "attack_success_rate": 0.X,
    "benign_clean_accuracy_rouge_l": 0.Y,
    "benign_answer_faithfulness": 0.Z,
    ...
  },
  "conflict_injection": {...},
  "soft_content_injection": {...}
}
```

## Key Features Implemented

| Feature | Status | Details |
|---------|--------|---------|
| Trigger-token poisoning | ✅ | Injects "cf" at 2-3 positions per doc |
| Conflict-injection | ✅ | 12 built-in contradiction pairs |
| Soft-content injection | ✅ | Off-topic + semantic keywords |
| Corpus loading | ✅ | 2067 docs from SQuAD |
| Evaluation framework | ✅ | ASR, ROUGE-L, faithfulness metrics |
| RFC integration | ✅ | Query-time defense ready |
| EllipticEnvelope baseline | ✅ | Ingestion-time defense ready |

## Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'faiss'`
```bash
# Solution: Make sure conda environment is activated
conda activate ragdefense
python scripts/validate_attacks.py
```

**Problem:** CUDA/GPU issues
```bash
# Solution: Use CPU versions
python scripts/validate_attacks.py                    # Always works
python scripts/run_attacks_vs_defenses.py --no-llm    # CPU-friendly
```

**Problem:** Out of memory
```bash
# Solution: Reduce sample size
python scripts/run_attacks_vs_defenses.py --samples 10 --no-llm
```

## Next Phase

After evaluation results are ready:

1. **Fine-tuning module** (`src/finetune/__init__.py`)
   - QLoRA backdoor injection into LLaMA-3-8B
   - Combine corpus + model-level attacks
   - Expected: 2-3 hours implementation

2. **Mechanistic analysis**
   - PCA visualization of embedding space
   - Show when/why RFC catches/misses poisoned docs
   - Expected: 1-2 hours

3. **Final report**
   - RFC vs EllipticEnvelope comparison
   - Attack success rates across variants
   - Defense effectiveness metrics

---

**Total Attack Module Implementation: ✅ COMPLETE**  
**Status:** Ready for evaluation. Wait for environment build to finish, then run evaluation scripts.
