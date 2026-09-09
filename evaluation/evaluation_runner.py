"""Shared GSM8K loading, scoring, checkpointing, and result serialization."""

import json
from datetime import datetime, timezone
from pathlib import Path

from evaluator import (
    EVALUATOR_VERSION,
    check_answer,
    extract_gold,
    extract_prediction,
    is_format_compliant,
)
from prompts import prompt_metadata


SCHEMA_VERSION = "1.0"


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_gsm8k(data_path, num_samples, start_index=1):
    if start_index < 1:
        raise ValueError("start_index is one-based and must be >= 1")
    items = []
    with Path(data_path).open("r", encoding="utf-8") as f:
        for dataset_index, line in enumerate(f, start=1):
            if dataset_index < start_index:
                continue
            if len(items) >= num_samples:
                break
            item = json.loads(line)
            item["dataset_index"] = dataset_index
            items.append(item)
    return items


def _metrics(results):
    total = len(results)
    correct = sum(item["status"] == "correct" for item in results)
    wrong = sum(item["status"] == "wrong" for item in results)
    invalid = sum(item["status"] == "invalid" for item in results)
    format_compliant = sum(item.get("format_compliant", False) for item in results)
    valid = total - invalid
    return {
        "num_samples": total,
        "correct": correct,
        "wrong": wrong,
        "invalid": invalid,
        "accuracy": correct / total if total else 0.0,
        "invalid_rate": invalid / total if total else 0.0,
        "valid_accuracy": correct / valid if valid else 0.0,
        "format_compliant": format_compliant,
        "format_compliance_rate": format_compliant / total if total else 0.0,
    }


def _build_output(config, results, started_at, completed_at=None, run_status="running"):
    return {
        "schema_version": SCHEMA_VERSION,
        "run_status": run_status,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "model": config["model"],
        "backend": config["backend"],
        "dataset": {
            "name": "GSM8K",
            "split": "test",
            "selection": "contiguous_range",
            "start_index": config["start_index"],
            "end_index": config["start_index"] + config["num_samples"] - 1,
            "requested_samples": config["num_samples"],
            "dataset_indices": config["dataset_indices"],
        },
        "prompt": prompt_metadata(config["prompt_name"]),
        "evaluator": {"version": EVALUATOR_VERSION},
        "decoding": config["decoding"],
        "execution": config["execution"],
        **_metrics(results),
        "results": results,
    }


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _checkpoint_matches(payload, config):
    return (
        payload.get("model") == config["model"]
        and payload.get("backend") == config["backend"]
        and payload.get("prompt", {}).get("name") == config["prompt_name"]
        and payload.get("prompt", {}).get("version")
        == prompt_metadata(config["prompt_name"])["version"]
        and payload.get("dataset", {}).get("dataset_indices")
        == config["dataset_indices"]
        and payload.get("decoding") == config["decoding"]
        and payload.get("evaluator", {}).get("version") == EVALUATOR_VERSION
    )


def run_evaluation(
    *, ask_batch, model_name, backend_name, data_path, output_path,
    num_samples, batch_size, prompt_name, max_new_tokens,
    decoding, execution, start_index=1,
):
    """Run one controlled experiment and save a resumable checkpoint."""
    items = load_gsm8k(data_path, num_samples, start_index)
    if len(items) != num_samples:
        raise ValueError(f"Requested {num_samples} samples, found {len(items)}")

    output_path = Path(output_path)
    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    config = {
        "model": model_name,
        "backend": backend_name,
        "num_samples": num_samples,
        "start_index": start_index,
        "dataset_indices": [item["dataset_index"] for item in items],
        "prompt_name": prompt_name,
        "decoding": decoding,
        "execution": execution,
    }

    results = []
    started_at = _utc_now()
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if not _checkpoint_matches(checkpoint, config):
            raise RuntimeError(
                f"Checkpoint config does not match this run: {checkpoint_path}"
            )
        results = checkpoint["results"]
        started_at = checkpoint["started_at_utc"]
        print(f"Resuming from {len(results)}/{num_samples} completed samples.")

    completed = len(results)
    if completed > num_samples:
        raise RuntimeError("Checkpoint contains more results than requested")

    print(f"Loaded {len(items)} questions; starting at sample {completed + 1}.")

    try:
        for start in range(completed, num_samples, batch_size):
            batch_items = items[start : start + batch_size]
            questions = [item["question"] for item in batch_items]
            print(f"\nRunning samples {start + 1}-{start + len(batch_items)}...")
            responses = ask_batch(
                questions,
                max_new_tokens=max_new_tokens,
                prompt_name=prompt_name,
            )
            if len(responses) != len(batch_items):
                raise RuntimeError("Backend returned a different number of responses")

            for item, response in zip(batch_items, responses):
                gold = extract_gold(item["answer"])
                prediction = extract_prediction(response)
                if prediction is None:
                    status = "invalid"
                elif check_answer(prediction, gold):
                    status = "correct"
                else:
                    status = "wrong"

                result = {
                    "id": item["dataset_index"],
                    "question": item["question"],
                    "gold": gold,
                    "prediction": prediction,
                    "status": status,
                    "raw_response": response,
                    "format_compliant": is_format_compliant(response),
                }
                results.append(result)
                print(
                    f"[{result['id']:03d}] gold={gold} | "
                    f"pred={prediction} | {status}"
                )

            _write_json(
                checkpoint_path,
                _build_output(config, results, started_at),
            )

    except Exception as exc:
        failed = _build_output(config, results, started_at, run_status="failed")
        failed["backend_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "completed_samples": len(results),
        }
        _write_json(checkpoint_path, failed)
        raise RuntimeError(
            f"Evaluation stopped after {len(results)} samples. "
            f"Checkpoint saved to {checkpoint_path}"
        ) from exc

    output = _build_output(
        config, results, started_at,
        completed_at=_utc_now(), run_status="completed",
    )
    _write_json(output_path, output)
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print("\n" + "=" * 60)
    print(f"Total: {output['num_samples']}")
    print(f"Correct: {output['correct']}")
    print(f"Wrong: {output['wrong']}")
    print(f"Invalid: {output['invalid']}")
    print(f"Accuracy: {output['accuracy']:.2%}")
    print(f"Invalid Rate: {output['invalid_rate']:.2%}")
    print(f"Format Compliance: {output['format_compliance_rate']:.2%}")
    print("=" * 60)
    print(f"Results saved to: {output_path}")
    return output
