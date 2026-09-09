"""Create reproducible paired and error-taxonomy analyses for final holdout."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "evaluation" / "results"
OUTPUT = RESULTS / "final_holdout_error_analysis_i101-300.json"

FILES = {
    "base_a": "qwen_0.5b_gsm8k_prompt_a_n200_i101-300.json",
    "base_c": "qwen_0.5b_gsm8k_prompt_c_n200_i101-300.json",
    "random_a": "qwen_0.5b_random_sft_gsm8k_prompt_a_n200_i101-300.json",
    "random_c": "qwen_0.5b_random_sft_gsm8k_prompt_c_n200_i101-300.json",
    "error_guided_a": "qwen_0.5b_error_guided_sft_gsm8k_prompt_a_n200_i101-300.json",
    "error_guided_c": "qwen_0.5b_error_guided_sft_gsm8k_prompt_c_n200_i101-300.json",
}

CONTRASTS = (
    ("prompt_effect_base", "base_a", "base_c"),
    ("random_sft_effect_prompt_a", "base_a", "random_a"),
    ("error_guided_sft_effect_prompt_a", "base_a", "error_guided_a"),
    ("selection_effect_prompt_a", "random_a", "error_guided_a"),
    ("prompt_effect_random_sft", "random_a", "random_c"),
    ("prompt_effect_error_guided_sft", "error_guided_a", "error_guided_c"),
)

REPRESENTATIVE_BASE_ERRORS = (101, 118, 143, 144, 158, 172, 250, 293)
REPRESENTATIVE_REGRESSIONS = (102, 113, 121, 129, 130, 192)


def _load_runs() -> dict[str, dict]:
    return {
        name: json.loads((RESULTS / filename).read_text(encoding="utf-8"))
        for name, filename in FILES.items()
    }


def _by_id(run: dict) -> dict[int, dict]:
    return {item["id"]: item for item in run["results"]}


def _pairwise_index() -> dict[tuple[str, str, str, str], dict]:
    comparison = json.loads(
        (RESULTS / "final_holdout_comparison_i101-300.json").read_text(encoding="utf-8")
    )
    index = {}
    for row in comparison["pairwise"]:
        key = (
            row["left_condition"], row["left_prompt"],
            row["right_condition"], row["right_prompt"],
        )
        index[key] = row
    return index


def _condition_prompt(run: dict) -> tuple[str, str]:
    return run.get("execution", {}).get("condition_slug", "base"), run["prompt"]["name"]


def analyze() -> dict:
    runs = _load_runs()
    records = {name: _by_id(run) for name, run in runs.items()}
    pairwise = _pairwise_index()
    annotations_payload = json.loads(
        (RESULTS / "qwen_0.5b_gsm8k_prompt_a_n200_i101-300_errors.json").read_text(encoding="utf-8")
    )
    annotations = {item["id"]: item for item in annotations_payload["errors"]}

    contrasts = []
    for label, left_name, right_name in CONTRASTS:
        left_condition, left_prompt = _condition_prompt(runs[left_name])
        right_condition, right_prompt = _condition_prompt(runs[right_name])
        row = pairwise[(left_condition, left_prompt, right_condition, right_prompt)]
        contrasts.append({
            "contrast": label,
            "left": left_name,
            "right": right_name,
            "left_accuracy": runs[left_name]["accuracy"],
            "right_accuracy": runs[right_name]["accuracy"],
            "accuracy_change_percentage_points": 100 * (runs[right_name]["accuracy"] - runs[left_name]["accuracy"]),
            "left_correct_to_right_not_correct": row["left_only_correct"],
            "left_not_correct_to_right_correct": row["right_only_correct"],
            "net_correct_change": row["right_only_correct"] - row["left_only_correct"],
            "mcnemar_exact_p_value": row["mcnemar_exact_p_value"],
        })

    distribution = Counter(item["primary_error_type"] for item in annotations.values())
    taxonomy_rows = []
    alternative_names = ("base_c", "random_a", "random_c", "error_guided_a", "error_guided_c")
    for error_type in annotations_payload["taxonomy"]:
        ids = [item_id for item_id, item in annotations.items() if item["primary_error_type"] == error_type]
        row = {
            "error_type": error_type,
            "base_a_failures": len(ids),
            "rate_among_base_a_failures": len(ids) / len(annotations),
            "rate_among_all_holdout_samples": len(ids) / runs["base_a"]["num_samples"],
            "recovered_to_correct": {},
        }
        for name in alternative_names:
            recovered = sum(records[name][item_id]["status"] == "correct" for item_id in ids)
            row["recovered_to_correct"][name] = {
                "count": recovered,
                "rate": recovered / len(ids) if ids else None,
            }
        taxonomy_rows.append(row)

    invalid_cases = []
    for item_id, item in records["base_a"].items():
        if item["status"] == "invalid":
            annotation = annotations[item_id]
            invalid_cases.append({
                "id": item_id,
                "gold": item["gold"],
                "primary_error_type": annotation["primary_error_type"],
                "secondary_error_type": annotation.get("secondary_error_type"),
                "evidence": annotation["evidence"],
            })

    def example(item_id: int, include_annotation: bool) -> dict:
        base_item = records["base_a"][item_id]
        result = {
            "id": item_id,
            "question": base_item["question"],
            "gold": base_item["gold"],
            "outcomes": {
                name: {
                    "prediction": records[name][item_id]["prediction"],
                    "status": records[name][item_id]["status"],
                }
                for name in FILES
            },
        }
        if include_annotation:
            annotation = annotations[item_id]
            result["primary_error_type"] = annotation["primary_error_type"]
            result["secondary_error_type"] = annotation.get("secondary_error_type")
            result["evidence"] = annotation["evidence"]
        return result

    payload = {
        "schema_version": "1.0",
        "scope": "Frozen GSM8K test IDs 101-300; no post-hoc retraining permitted",
        "annotation_method": annotations_payload.get("annotation_method"),
        "annotation_audit": {
            "total_base_a_failures": len(annotations),
            "second_pass_reviewed": sum(item.get("review_status") == "second_pass_reviewed" for item in annotations.values()),
            "assistant_corrected_after_second_pass": sum(item.get("review_status") == "assistant_audited_needs_human_review" for item in annotations.values()),
            "human_review_required": True,
        },
        "primary_contrasts": contrasts,
        "base_a_error_taxonomy": taxonomy_rows,
        "base_a_invalid_cases": invalid_cases,
        "representative_base_errors": [example(item_id, True) for item_id in REPRESENTATIVE_BASE_ERRORS],
        "representative_sft_regressions": [example(item_id, False) for item_id in REPRESENTATIVE_REGRESSIONS],
        "main_findings": [
            "Both SFT conditions nearly eliminate invalid outputs and raise strict answer-format compliance to about 99%.",
            "Neither SFT condition improves holdout mathematical accuracy over Base under Prompt A.",
            "Error-guided SFT is not statistically distinguishable from Random SFT under Prompt A.",
            "Prompt C does not significantly improve accuracy within any of the three model conditions.",
            "Relation modeling is the dominant preliminary Base Prompt A failure category.",
        ],
        "limitations": [
            "Taxonomy labels are two-pass LLM-assisted preliminary annotations, with a targeted consistency audit; full human review is still required.",
            "McNemar p-values are unadjusted and should be interpreted with effect sizes and transition counts.",
            "The holdout has 200 examples, so small differences have wide uncertainty.",
            "Results apply to this model, prompts, 1,000-example one-epoch LoRA setup, and fixed holdout only.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    payload = analyze()
    print(f"saved={OUTPUT}")
    for row in payload["primary_contrasts"]:
        print(
            f"{row['contrast']}: {row['accuracy_change_percentage_points']:+.1f} pp, "
            f"transitions={row['left_correct_to_right_not_correct']}/{row['left_not_correct_to_right_correct']}, "
            f"p={row['mcnemar_exact_p_value']:.6g}"
        )
