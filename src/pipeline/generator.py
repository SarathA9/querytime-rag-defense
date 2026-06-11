import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Optional


class Generator:
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        load_in_4bit: bool = True,
        adapter_path: Optional[str] = None,
        device_map="auto",
    ):
        # Turing GPUs (RTX 2080 SUPER, T4) lack native bfloat16 — fall back to fp16.
        # Gate on compute capability >= 8.0 (Ampere+), not is_bf16_supported()
        # which counts slow Turing emulation.
        _bf16 = (
            torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        )
        dtype = torch.bfloat16 if _bf16 else torch.float16

        quant_cfg = None
        if load_in_4bit:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            device_map=device_map,
            torch_dtype=dtype,
        )

        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def _build_prompt(self, query: str, context_docs: List[str]) -> str:
        context = "\n\n".join(
            f"[Document {i + 1}]\n{doc}" for i, doc in enumerate(context_docs)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the question using only "
                    "the provided documents. Be concise and factual."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self,
        query: str,
        context_docs: List[str],
        max_new_tokens: int = 200,
        temperature: float = 0.1,
    ) -> str:
        prompt = self._build_prompt(query, context_docs)
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
