import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "evaluation")]

from mine_base_failures import build_candidate_ids  # noqa: E402
from prepare_error_guided_sft import prepare  # noqa: E402


def test_error_guided_contract():
    config_dir = ROOT / "training" / "configs"
    random_config = json.loads((config_dir / "random_sft.json").read_text(encoding="utf-8"))
    guided_config = json.loads((config_dir / "error_guided_sft.json").read_text(encoding="utf-8"))
    controlled = [
        "base_model", "seed", "prompt_name", "max_seq_length", "train_samples",
        "validation_samples", "epochs", "learning_rate", "micro_batch_size",
        "gradient_accumulation_steps", "weight_decay", "warmup_steps",
        "max_grad_norm", "eval_steps", "save_steps", "logging_steps", "lora",
    ]
    for key in controlled:
        assert random_config[key] == guided_config[key], key

    manifest = json.loads(
        (ROOT / "training" / "manifests" / "random_sft_seed42.json").read_text(encoding="utf-8")
    )
    candidates_a = build_candidate_ids(guided_config, manifest)
    candidates_b = build_candidate_ids(guided_config, manifest)
    validation = set(manifest["validation"]["dataset_indices"])
    excluded = {item["dataset_index"] for item in manifest["excluded_train_samples"]}
    assert candidates_a == candidates_b
    assert len(candidates_a) == 7372
    assert len(candidates_a) == len(set(candidates_a))
    assert not set(candidates_a) & validation
    assert not set(candidates_a) & excluded
    assert guided_config["mining"]["include_statuses"] == ["wrong", "invalid"]
    assert guided_config["mining"]["target_failures"] == 1000


def test_smoke_mining_to_sft():
    smoke = ROOT / "training" / "mining" / "base_failures_smoke_n2.json"
    if not smoke.exists():
        return
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        config = json.loads(
            (ROOT / "training" / "configs" / "error_guided_sft.json").read_text(
                encoding="utf-8"
            )
        )
        config["train_samples"] = 2
        config["train_data_path"] = str(temp / "train.jsonl")
        config["manifest_path"] = str(temp / "manifest.json")
        config_path = temp / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        manifest = prepare(config_path, smoke)
        rows = [
            json.loads(line)
            for line in (temp / "train.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(rows) == 2
        assert all(row["mined_status"] in {"wrong", "invalid"} for row in rows)
        assert all(row["assistant_answer"].count("####") == 1 for row in rows)
        assert manifest["error_guided_sft_train"]["count"] == 2
        assert manifest["validation"]["count"] == 100


if __name__ == "__main__":
    test_error_guided_contract()
    test_smoke_mining_to_sft()
    print("ALL PASSED")
