"""Prepare deterministic Random SFT train/validation and experiment manifests."""

import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = ROOT / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))

from prompts import PROMPT_VERSION, build_messages, prompt_metadata  # noqa: E402


CONFIG_PATH = ROOT / "training" / "configs" / "random_sft.json"
TRAIN_SOURCE = EVALUATION_DIR / "data" / "gsm8k" / "train.jsonl"
TEST_SOURCE = EVALUATION_DIR / "data" / "gsm8k" / "test.jsonl"
DATA_DIR = ROOT / "training" / "data"
MANIFEST_DIR = ROOT / "training" / "manifests"


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path):
    return _sha256_bytes(Path(path).read_bytes())


def _question_hash(question):
    normalized = " ".join(question.lower().split())
    return _sha256_bytes(normalized.encode("utf-8"))


def _load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def clean_answer(answer):
    reasoning, final = answer.rsplit("####", 1)
    reasoning = re.sub(r"<<[^<>]*>>", "", reasoning).strip()
    final = final.strip().replace(",", "")
    return f"{reasoning}\n#### {final}"


def _token_length(tokenizer, question, assistant_answer, prompt_name):
    messages = build_messages(question, prompt_name)
    messages.append({"role": "assistant", "content": assistant_answer})
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    return len(encoded["input_ids"])


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(config_path=CONFIG_PATH):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    train_source = _load_jsonl(TRAIN_SOURCE)
    test_source = _load_jsonl(TEST_SOURCE)
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        local_files_only=True,
    )

    train_question_hashes = {_question_hash(row["question"]) for row in train_source}
    test_question_hashes = {_question_hash(row["question"]) for row in test_source}
    overlap = train_question_hashes & test_question_hashes
    if overlap:
        raise RuntimeError(f"Found {len(overlap)} exact train/test question overlaps")

    eligible = []
    excluded = []
    for dataset_index, row in enumerate(train_source, start=1):
        assistant_answer = clean_answer(row["answer"])
        token_length = _token_length(
            tokenizer,
            row["question"],
            assistant_answer,
            config["prompt_name"],
        )
        prepared = {
            "dataset_index": dataset_index,
            "question": row["question"],
            "assistant_answer": assistant_answer,
            "token_length": token_length,
            "question_sha256": _question_hash(row["question"]),
        }
        if token_length <= config["max_seq_length"]:
            eligible.append(prepared)
        else:
            excluded.append({
                "dataset_index": dataset_index,
                "token_length": token_length,
                "reason": "formatted sequence exceeds max_seq_length",
            })

    rng = random.Random(config["seed"])
    rng.shuffle(eligible)
    validation_count = config["validation_samples"]
    train_count = config["train_samples"]
    validation_rows = eligible[:validation_count]
    train_rows = eligible[validation_count : validation_count + train_count]
    if len(train_rows) != train_count or len(validation_rows) != validation_count:
        raise RuntimeError("Not enough eligible GSM8K train examples")

    train_indices = {row["dataset_index"] for row in train_rows}
    validation_indices = {row["dataset_index"] for row in validation_rows}
    if train_indices & validation_indices:
        raise RuntimeError("Train and validation selections overlap")

    train_path = DATA_DIR / "random_sft_train_seed42_n1000.jsonl"
    validation_path = DATA_DIR / "random_sft_validation_seed42_n100.jsonl"
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "schema_version": "1.0",
        "source": {
            "train_path": str(TRAIN_SOURCE.resolve()),
            "train_sha256": _file_sha256(TRAIN_SOURCE),
            "test_path": str(TEST_SOURCE.resolve()),
            "test_sha256": _file_sha256(TEST_SOURCE),
        },
        "prompt": prompt_metadata(config["prompt_name"]),
        "seed": config["seed"],
        "max_seq_length": config["max_seq_length"],
        "eligible_train_samples": len(eligible),
        "excluded_train_samples": excluded,
        "validation": {
            "source_split": "train",
            "count": len(validation_rows),
            "dataset_indices": sorted(validation_indices),
            "jsonl_path": str(validation_path.resolve()),
            "jsonl_sha256": _file_sha256(validation_path),
        },
        "random_sft_train": {
            "source_split": "train",
            "count": len(train_rows),
            "dataset_indices": sorted(train_indices),
            "jsonl_path": str(train_path.resolve()),
            "jsonl_sha256": _file_sha256(train_path),
        },
        "diagnostic_dev": {
            "source_split": "test",
            "dataset_indices": list(range(1, 101)),
            "role": "prompt development and error diagnosis",
        },
        "final_holdout": {
            "source_split": "test",
            "dataset_indices": list(range(101, 301)),
            "role": "sealed final comparison; do not use for intervention design",
        },
        "train_test_exact_question_overlap": 0,
        "prompt_version": PROMPT_VERSION,
    }
    manifest_path = MANIFEST_DIR / "random_sft_seed42.json"
    manifest_path.write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Train samples: {len(train_rows)} -> {train_path}")
    print(f"Validation samples: {len(validation_rows)} -> {validation_path}")
    print(f"Excluded overlength: {len(excluded)}")
    print(f"Manifest: {manifest_path}")
    return split_manifest


if __name__ == "__main__":
    prepare()
