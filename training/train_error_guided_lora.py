"""Entry point for the failure-mined Error-guided GSM8K LoRA run."""

from train_lora import ROOT, run


CONFIG = ROOT / "training" / "configs" / "error_guided_sft.json"


if __name__ == "__main__":
    run(CONFIG)
