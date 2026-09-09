"""Re-score saved raw responses after an evaluator-only correction."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluator import (
    EVALUATOR_VERSION,
    check_answer,
    extract_prediction,
    is_format_compliant,
)


def rescore(path):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload["results"]
    previous_summary = {
        "evaluator": payload.get("evaluator"),
        "rescored_at_utc": payload.get("rescored_at_utc"),
        "correct": payload.get("correct"),
        "wrong": payload.get("wrong"),
        "invalid": payload.get("invalid"),
        "accuracy": payload.get("accuracy"),
        "invalid_rate": payload.get("invalid_rate"),
        "format_compliant": payload.get("format_compliant"),
        "format_compliance_rate": payload.get("format_compliance_rate"),
    }

    for item in results:
        prediction = extract_prediction(item["raw_response"])
        if prediction is None:
            status = "invalid"
        elif check_answer(prediction, item["gold"]):
            status = "correct"
        else:
            status = "wrong"
        item["prediction"] = prediction
        item["status"] = status
        item["format_compliant"] = is_format_compliant(item["raw_response"])

    total = len(results)
    correct = sum(item["status"] == "correct" for item in results)
    wrong = sum(item["status"] == "wrong" for item in results)
    invalid = sum(item["status"] == "invalid" for item in results)
    valid = total - invalid
    compliant = sum(item["format_compliant"] for item in results)

    payload.setdefault("rescore_history", []).append(previous_summary)
    payload.update({
        "evaluator": {"version": EVALUATOR_VERSION},
        "rescored_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_samples": total,
        "correct": correct,
        "wrong": wrong,
        "invalid": invalid,
        "accuracy": correct / total if total else 0.0,
        "invalid_rate": invalid / total if total else 0.0,
        "valid_accuracy": correct / valid if valid else 0.0,
        "format_compliant": compliant,
        "format_compliance_rate": compliant / total if total else 0.0,
    })
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_paths", nargs="+")
    args = parser.parse_args()
    for result_path in args.result_paths:
        payload = rescore(result_path)
        print(
            f"{result_path}: accuracy={payload['accuracy']:.2%}, "
            f"invalid={payload['invalid_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
