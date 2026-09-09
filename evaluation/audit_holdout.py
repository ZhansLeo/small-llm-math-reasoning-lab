"""Recompute and audit the frozen six-condition GSM8K holdout matrix.

This script treats every per-example record as the source of truth.  It verifies
alignment and parser decisions, recomputes aggregate metrics, checks the frozen
protocol hashes, and emits a compact machine-readable audit artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from evaluator import check_answer, extract_prediction, is_format_compliant


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
FROZEN = RESULTS / "final_protocol_frozen.json"
OUTPUT = RESULTS / "final_holdout_audit_i101-300.json"

RUNS = (
    ("base", "prompt_a", "qwen_0.5b_gsm8k_prompt_a_n200_i101-300.json"),
    ("base", "prompt_c", "qwen_0.5b_gsm8k_prompt_c_n200_i101-300.json"),
    ("random_sft", "prompt_a", "qwen_0.5b_random_sft_gsm8k_prompt_a_n200_i101-300.json"),
    ("random_sft", "prompt_c", "qwen_0.5b_random_sft_gsm8k_prompt_c_n200_i101-300.json"),
    ("error_guided_sft", "prompt_a", "qwen_0.5b_error_guided_sft_gsm8k_prompt_a_n200_i101-300.json"),
    ("error_guided_sft", "prompt_c", "qwen_0.5b_error_guided_sft_gsm8k_prompt_c_n200_i101-300.json"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - margin, center + margin]


def audit() -> dict:
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    expected_ids = frozen["holdout"]["indices"]
    expected_matrix = [(row["condition"], row["prompt"]) for row in frozen["matrix"]]
    configured_matrix = [(condition, prompt) for condition, prompt, _ in RUNS]
    checks = {
        "frozen_protocol_status": frozen.get("status") == "frozen",
        "matrix_matches_frozen_protocol": configured_matrix == expected_matrix,
        "artifact_hashes_match_frozen_protocol": True,
        "all_runs_completed": True,
        "all_ids_exactly_101_to_300_in_order": True,
        "questions_and_gold_align_across_runs": True,
        "stored_predictions_match_current_evaluator": True,
        "stored_statuses_match_recomputed_scores": True,
        "stored_format_flags_match_recomputed_flags": True,
        "stored_aggregates_match_records": True,
        "raw_responses_are_nonempty": True,
        "prompt_templates_consistent_within_prompt_name": True,
        "decoding_is_greedy_for_all_runs": True,
    }
    issues: list[str] = []

    for name, artifact in frozen["artifacts"].items():
        path = Path(artifact["path"])
        ok = path.is_file() and _sha256(path) == artifact["sha256"]
        if not ok:
            checks["artifact_hashes_match_frozen_protocol"] = False
            issues.append(f"Frozen artifact mismatch: {name}")

    reference_records = None
    prompt_templates: dict[str, str] = {}
    run_summaries = []
    for condition, prompt_name, filename in RUNS:
        path = RESULTS / filename
        run = json.loads(path.read_text(encoding="utf-8"))
        records = run["results"]
        ids = [item["id"] for item in records]
        if run.get("run_status") != "completed":
            checks["all_runs_completed"] = False
            issues.append(f"Run not completed: {filename}")
        if ids != expected_ids or len(set(ids)) != len(ids):
            checks["all_ids_exactly_101_to_300_in_order"] = False
            issues.append(f"ID mismatch: {filename}")

        if reference_records is None:
            reference_records = {item["id"]: (item["question"], item["gold"]) for item in records}
        else:
            for item in records:
                if reference_records.get(item["id"]) != (item["question"], item["gold"]):
                    checks["questions_and_gold_align_across_runs"] = False
                    issues.append(f"Question/gold mismatch: {filename}, id={item['id']}")

        template = run["prompt"]["template"]
        previous_template = prompt_templates.setdefault(prompt_name, template)
        if template != previous_template:
            checks["prompt_templates_consistent_within_prompt_name"] = False
            issues.append(f"Prompt template mismatch: {filename}")
        decoding = run.get("decoding", {})
        if decoding.get("policy") != "greedy" or decoding.get("do_sample") is not False:
            checks["decoding_is_greedy_for_all_runs"] = False
            issues.append(f"Non-greedy decoding: {filename}")

        counts = {"correct": 0, "wrong": 0, "invalid": 0}
        format_count = 0
        for item in records:
            raw = item.get("raw_response")
            if not isinstance(raw, str) or not raw.strip():
                checks["raw_responses_are_nonempty"] = False
                issues.append(f"Empty raw response: {filename}, id={item['id']}")
                continue
            prediction = extract_prediction(raw)
            if prediction != item.get("prediction"):
                checks["stored_predictions_match_current_evaluator"] = False
                issues.append(f"Prediction mismatch: {filename}, id={item['id']}")
            status = "invalid" if prediction is None else ("correct" if check_answer(prediction, item["gold"]) else "wrong")
            if status != item.get("status"):
                checks["stored_statuses_match_recomputed_scores"] = False
                issues.append(f"Status mismatch: {filename}, id={item['id']}")
            format_ok = is_format_compliant(raw)
            if format_ok != item.get("format_compliant"):
                checks["stored_format_flags_match_recomputed_flags"] = False
                issues.append(f"Format mismatch: {filename}, id={item['id']}")
            counts[status] += 1
            format_count += int(format_ok)

        aggregate_ok = (
            len(records) == run["num_samples"]
            and counts["correct"] == run["correct"]
            and counts["wrong"] == run["wrong"]
            and counts["invalid"] == run["invalid"]
            and format_count == run["format_compliant"]
            and math.isclose(counts["correct"] / len(records), run["accuracy"])
            and math.isclose(counts["invalid"] / len(records), run["invalid_rate"])
        )
        if not aggregate_ok:
            checks["stored_aggregates_match_records"] = False
            issues.append(f"Aggregate mismatch: {filename}")

        run_summaries.append({
            "condition": condition,
            "prompt": prompt_name,
            "n": len(records),
            **counts,
            "accuracy": counts["correct"] / len(records),
            "accuracy_wilson_95ci": _wilson(counts["correct"], len(records)),
            "invalid_rate": counts["invalid"] / len(records),
            "format_compliance_rate": format_count / len(records),
        })

    payload = {
        "schema_version": "1.0",
        "source_protocol": str(FROZEN.resolve()),
        "audit_passed": all(checks.values()),
        "checks": checks,
        "issues": issues,
        "runs": run_summaries,
        "interpretation_guardrails": [
            "Wilson intervals describe uncertainty for each marginal accuracy; paired model differences use exact McNemar tests.",
            "Failure to reject the McNemar null is not evidence that two conditions are equivalent.",
            "Holdout results must not be used to retune and retest on IDs 101-300.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = audit()
    print(f"audit_passed={result['audit_passed']}")
    print(f"issues={len(result['issues'])}")
    print(f"saved={OUTPUT}")
