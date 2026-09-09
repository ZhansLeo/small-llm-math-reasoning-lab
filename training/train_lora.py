"""Shared reproducible CPU LoRA trainer for controlled GSM8K SFT runs."""

import argparse
import json
import os
import platform
import shutil
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import accelerate
import datasets
import peft
import psutil
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from sft_data import GSM8KSFTDataset, SFTDataCollator


ROOT = Path(__file__).resolve().parent.parent


class ResourceMonitor:
    def __init__(self, interval_seconds=0.2):
        self.interval_seconds = interval_seconds
        self.process = psutil.Process()
        self.peak_rss = 0
        self.minimum_available = psutil.virtual_memory().available
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
            self.minimum_available = min(self.minimum_available, psutil.virtual_memory().available)
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._stop.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _environment_versions():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "datasets": datasets.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def parse_args(default_config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def run(default_config):
    args = parse_args(default_config)
    config_path = _resolve(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_path = _resolve(config["train_data_path"])
    validation_path = _resolve(config["validation_data_path"])
    manifest_path = _resolve(config["manifest_path"])
    for path in (train_path, validation_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Required prepared artifact not found: {path}")

    sys.path.insert(0, str(ROOT / "evaluation"))
    from prompts import build_messages

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_dataset = GSM8KSFTDataset(
        train_path, tokenizer, build_messages, config["prompt_name"], config["max_seq_length"]
    )
    validation_dataset = GSM8KSFTDataset(
        validation_path, tokenizer, build_messages, config["prompt_name"], config["max_seq_length"]
    )
    if args.smoke:
        train_dataset.rows = train_dataset.rows[:16]
        validation_dataset.rows = validation_dataset.rows[:8]

    print("Loading base model in float32 on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"], dtype=torch.float32, local_files_only=True
    )
    model.config.use_cache = False
    lora = config["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        bias=lora["bias"], target_modules=lora["target_modules"], task_type="CAUSAL_LM",
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,}/{total:,} ({100 * trainable / total:.4f}%)")

    run_name = config["smoke_run_name"] if args.smoke else config["run_name"]
    output_dir = ROOT / "training" / "outputs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "training_config.json")
    shutil.copy2(manifest_path, output_dir / "data_manifest.json")
    _save_json(output_dir / "environment_versions.json", _environment_versions())

    common = dict(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=config["micro_batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=config["learning_rate"], lr_scheduler_type="linear",
        warmup_steps=config["warmup_steps"], weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"], optim="adamw_torch", use_cpu=True,
        bf16=False, fp16=False, gradient_checkpointing=args.gradient_checkpointing,
        use_cache=False, dataloader_num_workers=0, dataloader_pin_memory=False,
        remove_unused_columns=False, logging_strategy="steps",
        logging_steps=1 if args.smoke else config["logging_steps"], logging_first_step=True,
        report_to="none", seed=config["seed"], data_seed=config["seed"],
        prediction_loss_only=True,
    )
    if args.smoke:
        training_args = TrainingArguments(
            **common, max_steps=2, eval_strategy="no", save_strategy="no"
        )
    else:
        training_args = TrainingArguments(
            **common, num_train_epochs=config["epochs"], eval_strategy="steps",
            eval_steps=config["eval_steps"], save_strategy="steps",
            save_steps=config["save_steps"], save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        eval_dataset=validation_dataset, data_collator=SFTDataCollator(tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    started = time.perf_counter()
    with ResourceMonitor() as monitor:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elapsed = time.perf_counter() - started

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    _save_json(output_dir / "resource_metrics.json", {
        "elapsed_seconds": elapsed,
        "peak_process_rss_gb": monitor.peak_rss / (2 ** 30),
        "minimum_system_available_gb": monitor.minimum_available / (2 ** 30),
        "trainable_parameters": trainable,
        "total_parameters_with_adapter": total,
        "trainable_percentage": 100 * trainable / total,
        "gradient_checkpointing": args.gradient_checkpointing,
        "smoke": args.smoke,
    })
    _save_json(output_dir / "train_metrics.json", train_result.metrics)
    _save_json(output_dir / "log_history.json", trainer.state.log_history)
    print(f"Adapter saved to: {adapter_dir}")
    print(f"Elapsed: {elapsed:.1f}s; peak RSS: {monitor.peak_rss / (2 ** 30):.2f} GB")
