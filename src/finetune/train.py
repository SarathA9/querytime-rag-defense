"""
QLoRA fine-tuning that produces the backdoored LLaMA-3-8B-Instruct adapter.

This is the weight-level half of the dual-surface threat in the study. It loads
the base model in 4-bit (NF4, double-quant, bf16 compute) — identical to the
deployed ``Generator`` — attaches a LoRA adapter, and trains on the poisoned
instruction set from ``dataset.build_examples``. The saved adapter is loaded at
deployment via ``Generator(adapter_path=...)``.

No ``trl`` dependency: training uses ``transformers.Trainer`` with a
completion-only loss collator, which is version-stable and gives a clean
backdoor signal.
"""

import os

# Force the PyTorch backend; this env has Keras 3 which breaks transformers' TF import.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from .dataset import BackdoorDataset, Example


# LoRA target modules for LLaMA-3: attention + MLP projections.
LLAMA_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class TrainConfig:
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    output_dir: str = "results/backdoor_adapter"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    # Single-GPU QLoRA: pin the whole model to one device. "auto" can try to
    # offload layers to CPU/disk, which 4-bit bitsandbytes refuses. Override to
    # "auto" only on multi-GPU nodes.
    device_map: object = field(default_factory=lambda: {"": 0})
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_strategy: str = "epoch"
    seed: int = 42


@dataclass
class CompletionCollator:
    """Pads input_ids/attention_mask/labels to the batch max length.

    Labels are padded with -100 so padding never contributes to the loss.
    """

    pad_token_id: int
    label_pad_id: int = -100

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for f in features:
            pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * pad)
            batch["attention_mask"].append(f["attention_mask"] + [0] * pad)
            batch["labels"].append(f["labels"] + [self.label_pad_id] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Right padding for training (left padding is only needed for generation).
    tok.padding_side = "right"
    return tok


def supports_bf16() -> bool:
    """True only on Ampere+ (compute capability >= 8.0). Turing cards
    (RTX 2080 SUPER, T4) have no native bfloat16 — they must use float16.

    NB: ``torch.cuda.is_bf16_supported()`` returns True on Turing via *emulation*,
    but ``TrainingArguments(bf16=True)`` uses the stricter Ampere-only check and
    raises on Turing. Gate on compute capability to keep the two consistent."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def compute_dtype() -> torch.dtype:
    return torch.bfloat16 if supports_bf16() else torch.float16


def build_qlora_model(cfg: TrainConfig):
    """Load the base model in 4-bit and attach a LoRA adapter."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    dtype = compute_dtype()
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=quant_cfg,
        device_map=cfg.device_map,
        torch_dtype=dtype,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=LLAMA_LORA_TARGETS,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def train_backdoor(
    examples: List[Example],
    cfg: Optional[TrainConfig] = None,
) -> str:
    """Fine-tune the backdoored adapter and save it.

    Args:
        examples: output of ``dataset.build_examples`` (clean + poisoned).
        cfg: training hyperparameters.

    Returns:
        Path to the saved LoRA adapter directory.
    """
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)

    tokenizer = load_tokenizer(cfg.model_name)
    model = build_qlora_model(cfg)
    train_ds = BackdoorDataset(tokenizer, examples, max_length=cfg.max_length)

    bf16 = supports_bf16()
    print(f"Mixed precision: {'bf16' if bf16 else 'fp16'} "
          f"(device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        bf16=bf16,
        fp16=not bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=CompletionCollator(pad_token_id=tokenizer.pad_token_id),
    )
    trainer.train()

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    return cfg.output_dir
