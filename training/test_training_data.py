import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "evaluation"), str(ROOT / "training")]

from prompts import build_messages  # noqa: E402
from sft_data import GSM8KSFTDataset, SFTDataCollator  # noqa: E402


def test_training_data():
    train_path = ROOT / "training" / "data" / "random_sft_train_seed42_n1000.jsonl"
    validation_path = (
        ROOT / "training" / "data" / "random_sft_validation_seed42_n100.jsonl"
    )
    manifest = json.loads(
        (ROOT / "training" / "manifests" / "random_sft_seed42.json").read_text(
            encoding="utf-8"
        )
    )
    train_ids = set(manifest["random_sft_train"]["dataset_indices"])
    validation_ids = set(manifest["validation"]["dataset_indices"])
    holdout_ids = set(manifest["final_holdout"]["dataset_indices"])
    assert len(train_ids) == 1000
    assert len(validation_ids) == 100
    assert not train_ids & validation_ids
    assert holdout_ids == set(range(101, 301))
    assert manifest["train_test_exact_question_overlap"] == 0

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    train = GSM8KSFTDataset(
        train_path,
        tokenizer,
        build_messages,
        "prompt_a",
        512,
    )
    validation = GSM8KSFTDataset(
        validation_path,
        tokenizer,
        build_messages,
        "prompt_a",
        512,
    )
    assert len(train) == 1000
    assert len(validation) == 100

    examples = [train[0], train[1]]
    for example in examples:
        first_target = next(
            index for index, label in enumerate(example["labels"]) if label != -100
        )
        assert first_target > 0
        assert all(label == -100 for label in example["labels"][:first_target])
        assert all(label != -100 for label in example["labels"][first_target:])
        assert len(example["input_ids"]) <= 512

    batch = SFTDataCollator(tokenizer.pad_token_id)(examples)
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["input_ids"].shape == batch["attention_mask"].shape


if __name__ == "__main__":
    test_training_data()
    print("ALL PASSED")
