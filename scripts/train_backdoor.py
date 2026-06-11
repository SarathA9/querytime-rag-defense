"""
Produce the backdoored LLaMA-3-8B-Instruct adapter (weight-level attack surface).

Builds a poisoned instruction set (clean RAG-QA + trigger->target examples) from
the NQ samples, then QLoRA fine-tunes a LoRA adapter that is loaded at deployment
via ``Generator(adapter_path=...)``.

Usage:
    # Inspect the dataset without a GPU / without training:
    python scripts/train_backdoor.py --dry-run

    # Train (requires GPU + access to meta-llama/Meta-Llama-3-8B-Instruct):
    python scripts/train_backdoor.py --epochs 3 --output results/backdoor_adapter
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.attacks import load_corpus, load_samples
from src.finetune import build_examples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--samples", type=int, default=None, help="Cap on training samples")
    p.add_argument("--output", default=None, help="Adapter output dir")
    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--poison-rate", type=float, default=None)
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Token cap per example. Lower (256-512) to fit tight VRAM.",
    )
    p.add_argument(
        "--device-map",
        default=None,
        help="'single' pins to GPU 0 (default, best for one GPU); 'auto' shards "
        "across multiple GPUs.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build + print dataset examples; skip model load and training.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    ft = cfg.get("finetune", {})
    atk = cfg.get("attacks", {}).get("trigger_token", {})
    data_dir = ROOT / "data"

    trigger = atk.get("trigger", "cf")
    target_output = atk.get("target_output", "I cannot answer this question.")
    poison_rate = args.poison_rate if args.poison_rate is not None else ft.get("poison_rate", 0.5)
    n_distractors = ft.get("n_distractors", 2)
    seed = ft.get("seed", 42)

    samples = load_samples(data_dir / "nq_samples.json")
    if args.samples:
        samples = samples[: args.samples]
    corpus = load_corpus(data_dir / "corpus.json")

    examples = build_examples(
        samples,
        corpus=corpus,
        trigger=trigger,
        target_output=target_output,
        poison_rate=poison_rate,
        n_distractors=n_distractors,
        seed=seed,
    )

    n_pois = sum(e.poisoned for e in examples)
    print(f"Built {len(examples)} examples | poisoned={n_pois} clean={len(examples) - n_pois}")
    print(f"Trigger={trigger!r}  Target={target_output!r}  poison_rate={poison_rate}")

    if args.dry_run:
        print("\n--- sample poisoned example ---")
        ex = next(e for e in examples if e.poisoned)
        print("Q:", ex.question)
        print("#context docs:", len(ex.context_docs))
        print("Response:", ex.response)
        print("\n--- sample clean example ---")
        ex = next(e for e in examples if not e.poisoned)
        print("Q:", ex.question)
        print("#context docs:", len(ex.context_docs))
        print("Response:", ex.response)
        print("\n(dry run: no model loaded, no training performed)")
        return

    # Heavy imports only when actually training.
    from src.finetune import TrainConfig, train_backdoor

    max_length = args.max_length if args.max_length is not None else ft.get("max_length", 2048)
    device_map_arg = args.device_map or ft.get("device_map", "single")
    device_map = "auto" if device_map_arg == "auto" else {"": 0}

    train_cfg = TrainConfig(
        model_name=cfg.get("generator", {}).get("model_name", TrainConfig.model_name),
        output_dir=args.output or ft.get("output_dir", str(ROOT / "results" / "backdoor_adapter")),
        lora_r=ft.get("lora_r", 16),
        lora_alpha=ft.get("lora_alpha", 32),
        lora_dropout=ft.get("lora_dropout", 0.05),
        learning_rate=ft.get("learning_rate", 2e-4),
        num_train_epochs=args.epochs if args.epochs is not None else ft.get("epochs", 3.0),
        per_device_train_batch_size=ft.get("batch_size", 1),
        gradient_accumulation_steps=ft.get("grad_accum", 8),
        max_length=max_length,
        device_map=device_map,
        seed=seed,
    )
    out = train_backdoor(examples, train_cfg)
    print(f"\n✓ Backdoored adapter saved to {out}")
    print(f"  Deploy with: Generator(adapter_path='{out}')")


if __name__ == "__main__":
    main()
