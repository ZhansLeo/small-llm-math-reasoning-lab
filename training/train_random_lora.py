"""Backward-compatible entry point for the Random GSM8K LoRA run."""

from prepare_random_sft import prepare
from train_lora import ROOT, run


CONFIG = ROOT / "training" / "configs" / "random_sft.json"


if __name__ == "__main__":
    data = ROOT / "training" / "data" / "random_sft_train_seed42_n1000.jsonl"
    if not data.exists():
        prepare(CONFIG)
    run(CONFIG)
