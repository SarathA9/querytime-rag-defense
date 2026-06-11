"""Fine-tuning (weight-level) attack surface: build a backdoored LLaMA-3-8B
adapter via QLoRA on a poisoned instruction set.

``dataset`` is light (no torch/transformers) and imported eagerly so dataset
construction / dry-runs work without the heavy ML stack. ``train`` is imported
lazily so importing this package never pulls in transformers unless training.
"""

from .dataset import (
    Example,
    BackdoorDataset,
    build_examples,
    render_prompt,
    tokenize_example,
    SYSTEM_PROMPT,
)

__all__ = [
    "Example",
    "BackdoorDataset",
    "build_examples",
    "render_prompt",
    "tokenize_example",
    "SYSTEM_PROMPT",
    "TrainConfig",
    "train_backdoor",
    "build_qlora_model",
    "load_tokenizer",
]

_TRAIN_SYMBOLS = {
    "TrainConfig",
    "train_backdoor",
    "build_qlora_model",
    "load_tokenizer",
    "CompletionCollator",
}


def __getattr__(name):
    # Lazily forward training symbols so transformers is only imported on demand.
    if name in _TRAIN_SYMBOLS:
        from . import train

        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
