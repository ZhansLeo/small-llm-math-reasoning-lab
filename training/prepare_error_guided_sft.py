"""Build Error-guided SFT data and a reproducibility manifest from mining output."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from prepare_random_sft import (
    ROOT, TRAIN_SOURCE, TEST_SOURCE,
    _file_sha256, _load_jsonl, _question_hash, _token_length, _write_jsonl, clean_answer,
)


CONFIG = ROOT / "training" / "configs" / "error_guided_sft.json"
MINING_RESULT = ROOT / "training" / "mining" / "base_failures_seed42_n1000.json"
RANDOM_MANIFEST = ROOT / "training" / "manifests" / "random_sft_seed42.json"


def prepare(config_path=CONFIG, mining_path=MINING_RESULT):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    mining = json.loads(Path(mining_path).read_text(encoding="utf-8"))
    if mining.get("run_status") != "completed":
        raise RuntimeError("Failure mining must be completed before preparing SFT data")
    target = config["train_samples"]
    selected_ids = mining["selection"]["selected_ids"]
    if len(selected_ids) != target or len(set(selected_ids)) != target:
        raise RuntimeError("Mining result must contain exactly 1,000 unique selected IDs")

    random_manifest = json.loads(RANDOM_MANIFEST.read_text(encoding="utf-8"))
    validation_ids = set(random_manifest["validation"]["dataset_indices"])
    random_ids = set(random_manifest["random_sft_train"]["dataset_indices"])
    if validation_ids & set(selected_ids):
        raise RuntimeError("Error-guided train selection overlaps validation")

    source_rows = _load_jsonl(TRAIN_SOURCE)
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], local_files_only=True)
    prepared = []
    for dataset_index in selected_ids:
        source = source_rows[dataset_index - 1]
        assistant_answer = clean_answer(source["answer"])
        token_length = _token_length(
            tokenizer, source["question"], assistant_answer, config["prompt_name"]
        )
        if token_length > config["max_seq_length"]:
            raise RuntimeError(f"Selected example {dataset_index} exceeds max_seq_length")
        prepared.append({
            "dataset_index": dataset_index,
            "question": source["question"],
            "assistant_answer": assistant_answer,
            "token_length": token_length,
            "question_sha256": _question_hash(source["question"]),
            "mined_status": next(
                item["status"] for item in mining["results"] if item["id"] == dataset_index
            ),
        })

    output_path = ROOT / config["train_data_path"]
    _write_jsonl(output_path, prepared)
    manifest_path = ROOT / config["manifest_path"]
    manifest = {
        "schema_version": "1.0",
        "experiment_name": config["experiment_name"],
        "source": {
            "train_path": str(TRAIN_SOURCE.resolve()),
            "train_sha256": _file_sha256(TRAIN_SOURCE),
            "test_path": str(TEST_SOURCE.resolve()),
            "test_sha256": _file_sha256(TEST_SOURCE),
        },
        "selection": {
            "method": "base_model_failure_mining",
            "mining_result": str(Path(mining_path).resolve()),
            "mining_result_sha256": _file_sha256(mining_path),
            "contract_fingerprint": mining["contract_fingerprint"],
            "processed_candidates": mining["processed_candidates"],
            "candidate_counts": mining["counts"],
            "observed_failure_rate": mining["observed_failure_rate"],
            "include_statuses": mining["selection"]["include_statuses"],
        },
        "error_guided_sft_train": {
            "source_split": "train",
            "count": len(prepared),
            "dataset_indices_in_selection_order": selected_ids,
            "jsonl_path": str(output_path.resolve()),
            "jsonl_sha256": _file_sha256(output_path),
            "overlap_with_random_sft_count": len(set(selected_ids) & random_ids),
        },
        "validation": random_manifest["validation"],
        "diagnostic_dev": random_manifest["diagnostic_dev"],
        "final_holdout": random_manifest["final_holdout"],
        "train_test_exact_question_overlap": random_manifest["train_test_exact_question_overlap"],
        "prompt": random_manifest["prompt"],
        "seed": config["seed"],
        "max_seq_length": config["max_seq_length"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {len(prepared)} Error-guided examples -> {output_path}")
    print(f"Random SFT overlap: {manifest['error_guided_sft_train']['overlap_with_random_sft_count']}")
    print(f"Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    prepare()
