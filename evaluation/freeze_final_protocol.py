"""Freeze adapter/config hashes and the six-condition matrix before holdout access."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
OUTPUT = RESULTS / "final_protocol_frozen.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_dev(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("run_status") != "completed":
        raise RuntimeError(f"Development run is incomplete: {path}")
    if payload.get("dataset", {}).get("dataset_indices") != list(range(1, 101)):
        raise RuntimeError(f"Development run has wrong sample IDs: {path}")
    return payload


def freeze():
    random_adapter = ROOT / "training" / "outputs" / "random_lora_seed42_n1000" / "adapter" / "adapter_model.safetensors"
    error_adapter = ROOT / "training" / "outputs" / "error_guided_lora_seed42_n1000" / "adapter" / "adapter_model.safetensors"
    random_config = ROOT / "training" / "configs" / "random_sft.json"
    error_config = ROOT / "training" / "configs" / "error_guided_sft.json"
    dev_paths = [
        RESULTS / "qwen_0.5b_gsm8k_prompt_a_n100.json",
        RESULTS / "qwen_0.5b_gsm8k_prompt_c_n100.json",
        RESULTS / "qwen_0.5b_random_sft_gsm8k_prompt_a_n100.json",
        RESULTS / "qwen_0.5b_random_sft_gsm8k_prompt_c_n100.json",
        RESULTS / "qwen_0.5b_error_guided_sft_gsm8k_prompt_a_n100.json",
        RESULTS / "qwen_0.5b_error_guided_sft_gsm8k_prompt_c_n100.json",
    ]
    for path in dev_paths:
        _require_dev(path)
    payload = {
        "schema_version": "1.0",
        "status": "frozen",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout": {"dataset": "GSM8K", "split": "test", "indices": list(range(101, 301))},
        "matrix": [
            {"condition": condition, "prompt": prompt}
            for condition in ("base", "random_sft", "error_guided_sft")
            for prompt in ("prompt_a", "prompt_c")
        ],
        "artifacts": {
            "random_adapter": {"path": str(random_adapter.resolve()), "sha256": _sha256(random_adapter)},
            "error_guided_adapter": {"path": str(error_adapter.resolve()), "sha256": _sha256(error_adapter)},
            "random_config": {"path": str(random_config.resolve()), "sha256": _sha256(random_config)},
            "error_guided_config": {"path": str(error_config.resolve()), "sha256": _sha256(error_config)},
        },
        "development_results": [
            {"path": str(path.resolve()), "sha256": _sha256(path)} for path in dev_paths
        ],
        "policy": "Open indices 101-300 once; do not tune or retrain from holdout results.",
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Final protocol frozen: {OUTPUT}")
    return payload


if __name__ == "__main__":
    freeze()
