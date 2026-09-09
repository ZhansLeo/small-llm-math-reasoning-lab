"""Mine reproducible GSM8K train failures from the unadapted base model."""

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = ROOT / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))

from evaluator import (  # noqa: E402
    EVALUATOR_VERSION,
    check_answer,
    extract_gold,
    extract_prediction,
    is_format_compliant,
)
from inference import DEVICE, MODEL_NAME, ask_batch  # noqa: E402
from prompts import prompt_metadata  # noqa: E402


DEFAULT_CONFIG = ROOT / "training" / "configs" / "error_guided_sft.json"
TRAIN_SOURCE = EVALUATION_DIR / "data" / "gsm8k" / "train.jsonl"
RANDOM_MANIFEST = ROOT / "training" / "manifests" / "random_sft_seed42.json"
MINING_DIR = ROOT / "training" / "mining"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def build_candidate_ids(config, split_manifest):
    excluded = {item["dataset_index"] for item in split_manifest["excluded_train_samples"]}
    validation = set(split_manifest["validation"]["dataset_indices"])
    candidates = [
        index for index in range(1, split_manifest["eligible_train_samples"] + len(excluded) + 1)
        if index not in excluded and index not in validation
    ]
    random.Random(config["mining"]["seed"]).shuffle(candidates)
    return candidates


def _fingerprint(config, candidate_ids, target_failures):
    contract = {
        "base_model": config["base_model"],
        "mining": config["mining"],
        "target_failures": target_failures,
        "candidate_ids": candidate_ids,
        "train_sha256": _sha256(TRAIN_SOURCE),
        "evaluator_version": EVALUATOR_VERSION,
        "prompt": prompt_metadata(config["mining"]["prompt_name"]),
    }
    encoded = json.dumps(contract, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _payload(config, fingerprint, candidate_ids, results, target, started, status):
    included = set(config["mining"]["include_statuses"])
    selected = [item["id"] for item in results if item["status"] in included][:target]
    counts = {
        label: sum(item["status"] == label for item in results)
        for label in ("correct", "wrong", "invalid")
    }
    return {
        "schema_version": "1.0",
        "run_status": status,
        "started_at_utc": started,
        "completed_at_utc": _utc_now() if status == "completed" else None,
        "contract_fingerprint": fingerprint,
        "model": MODEL_NAME,
        "backend": "local_transformers",
        "device": str(DEVICE),
        "dataset": {"name": "GSM8K", "split": "train", "source_sha256": _sha256(TRAIN_SOURCE)},
        "candidate_order": {
            "method": "seeded_shuffle_after_validation_and_overlength_exclusion",
            "seed": config["mining"]["seed"],
            "total_candidates": len(candidate_ids),
        },
        "prompt": prompt_metadata(config["mining"]["prompt_name"]),
        "evaluator": {"version": EVALUATOR_VERSION},
        "decoding": {
            "policy": "greedy", "do_sample": False, "temperature": None,
            "max_new_tokens": config["mining"]["max_new_tokens"],
        },
        "batch_size": config["mining"]["batch_size"],
        "selection": {
            "method": config["mining"]["selection"],
            "include_statuses": sorted(included),
            "target_failures": target,
            "selected_ids": selected,
        },
        "processed_candidates": len(results),
        "counts": counts,
        "observed_failure_rate": len(selected) / len(results) if results else 0.0,
        "results": results,
    }


def mine(config_path=DEFAULT_CONFIG, output_path=None, target_override=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    split_manifest = json.loads(RANDOM_MANIFEST.read_text(encoding="utf-8"))
    source_rows = _load_jsonl(TRAIN_SOURCE)
    candidates = build_candidate_ids(config, split_manifest)
    target = target_override or config["mining"]["target_failures"]
    if target < 1:
        raise ValueError("target_failures must be positive")
    output_path = Path(output_path or MINING_DIR / f"base_failures_seed42_n{target}.json")
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    fingerprint = _fingerprint(config, candidates, target)
    if output_path.exists():
        completed = json.loads(output_path.read_text(encoding="utf-8"))
        if (
            completed.get("run_status") == "completed"
            and completed.get("contract_fingerprint") == fingerprint
            and len(completed.get("selection", {}).get("selected_ids", [])) == target
        ):
            print(f"Using completed mining result: {output_path}")
            return completed
        raise RuntimeError(f"Existing mining output does not match this contract: {output_path}")
    started = _utc_now()
    results = []
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("contract_fingerprint") != fingerprint:
            raise RuntimeError(f"Checkpoint contract mismatch: {checkpoint_path}")
        results = checkpoint["results"]
        started = checkpoint["started_at_utc"]
        expected = candidates[: len(results)]
        if [item["id"] for item in results] != expected:
            raise RuntimeError("Checkpoint candidate order is inconsistent")
        print(f"Resuming after {len(results)} candidates.")

    included = set(config["mining"]["include_statuses"])
    selected_count = sum(item["status"] in included for item in results)
    batch_size = config["mining"]["batch_size"]
    try:
        while selected_count < target and len(results) < len(candidates):
            ids = candidates[len(results) : len(results) + batch_size]
            questions = [source_rows[index - 1]["question"] for index in ids]
            responses = ask_batch(
                questions,
                max_new_tokens=config["mining"]["max_new_tokens"],
                prompt_name=config["mining"]["prompt_name"],
            )
            if len(responses) != len(ids):
                raise RuntimeError("Backend returned a different number of responses")
            for dataset_index, response in zip(ids, responses):
                source = source_rows[dataset_index - 1]
                gold = extract_gold(source["answer"])
                prediction = extract_prediction(response)
                status = "invalid" if prediction is None else (
                    "correct" if check_answer(prediction, gold) else "wrong"
                )
                results.append({
                    "id": dataset_index,
                    "question": source["question"],
                    "gold": gold,
                    "prediction": prediction,
                    "status": status,
                    "raw_response": response,
                    "format_compliant": is_format_compliant(response),
                })
                if status in included:
                    selected_count += 1
                print(
                    f"candidate={len(results):04d} id={dataset_index:04d} "
                    f"status={status} failures={selected_count}/{target}"
                )
                if selected_count >= target:
                    break
            _write_json_atomic(
                checkpoint_path,
                _payload(config, fingerprint, candidates, results, target, started, "running"),
            )
    except Exception as exc:
        failed = _payload(config, fingerprint, candidates, results, target, started, "failed")
        failed["backend_error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json_atomic(checkpoint_path, failed)
        raise

    if selected_count < target:
        raise RuntimeError(
            f"Candidate pool exhausted with only {selected_count}/{target} failures"
        )
    completed = _payload(config, fingerprint, candidates, results, target, started, "completed")
    _write_json_atomic(output_path, completed)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"Mining complete: {selected_count} failures from {len(results)} candidates")
    print(f"Saved to: {output_path}")
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=None)
    parser.add_argument("--target-failures", type=int, default=None)
    args = parser.parse_args()
    mine(args.config, args.output, args.target_failures)


if __name__ == "__main__":
    main()
