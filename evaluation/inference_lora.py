"""Local Qwen + PEFT adapter inference, independent from base inference.py."""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import build_messages


DEVICE = torch.device("cpu")


class LoRAInferenceBackend:
    def __init__(self, adapter_path):
        self.adapter_path = Path(adapter_path).resolve()
        if not self.adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter directory not found: {self.adapter_path}")
        self.tokenizer = None
        self.model = None
        self.base_model_name = None

    def _load(self):
        if self.model is not None:
            return self.tokenizer, self.model

        peft_config = PeftConfig.from_pretrained(
            self.adapter_path,
            local_files_only=True,
        )
        self.base_model_name = peft_config.base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            dtype=torch.float32,
            local_files_only=True,
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_path,
            local_files_only=True,
        )
        self.model.to(DEVICE)
        self.model.eval()
        return self.tokenizer, self.model

    @torch.no_grad()
    def ask_batch(self, questions, max_new_tokens=1024, prompt_name="prompt_a"):
        tokenizer, model = self._load()
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(question, prompt_name),
                add_generation_prompt=True,
                tokenize=False,
            )
            for question in questions
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        input_length = inputs["input_ids"].shape[1]
        return [
            tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
            for output in outputs
        ]
