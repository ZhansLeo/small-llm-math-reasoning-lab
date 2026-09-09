"""Compare controlled evaluation runs after verifying sample alignment."""

import argparse
import json
import math
from pathlib import Path


def compare(result_paths):
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in result_paths]
    if not runs:
        raise ValueError("At least one result file is required")

    expected_ids = [item["id"] for item in runs[0]["results"]]
    for path, run in zip(result_paths, runs):
        if run.get("run_status") != "completed":
            raise ValueError(f"Run is not completed: {path}")
        ids = [item["id"] for item in run["results"]]
        if ids != expected_ids:
            raise ValueError(f"Sample IDs do not align: {path}")

    rows = []
    for path, run in zip(result_paths, runs):
        condition = run.get("execution", {}).get("condition_slug", "base")
        rows.append({
            "result_file": str(Path(path).resolve()),
            "model": run["model"],
            "backend": run["backend"],
            "condition": condition,
            "prompt_name": run["prompt"]["name"],
            "prompt_version": run["prompt"]["version"],
            "num_samples": run["num_samples"],
            "correct": run["correct"],
            "wrong": run["wrong"],
            "invalid": run["invalid"],
            "accuracy": run["accuracy"],
            "invalid_rate": run["invalid_rate"],
            "format_compliance_rate": run.get("format_compliance_rate"),
        })
    pairwise = []
    for left_index in range(len(runs)):
        for right_index in range(left_index + 1, len(runs)):
            left = runs[left_index]
            right = runs[right_index]
            left_status = {item["id"]: item["status"] for item in left["results"]}
            right_status = {item["id"]: item["status"] for item in right["results"]}
            left_only = sum(
                left_status[item_id] == "correct" and right_status[item_id] != "correct"
                for item_id in expected_ids
            )
            right_only = sum(
                left_status[item_id] != "correct" and right_status[item_id] == "correct"
                for item_id in expected_ids
            )
            both_correct = sum(
                left_status[item_id] == right_status[item_id] == "correct"
                for item_id in expected_ids
            )
            discordant = left_only + right_only
            if discordant:
                tail = sum(
                    math.comb(discordant, k)
                    for k in range(min(left_only, right_only) + 1)
                ) / (2 ** discordant)
                p_value = min(1.0, 2 * tail)
            else:
                p_value = 1.0
            pairwise.append({
                "left_model": left["model"],
                "left_condition": left.get("execution", {}).get("condition_slug", "base"),
                "left_prompt": left["prompt"]["name"],
                "right_model": right["model"],
                "right_condition": right.get("execution", {}).get("condition_slug", "base"),
                "right_prompt": right["prompt"]["name"],
                "both_correct": both_correct,
                "left_only_correct": left_only,
                "right_only_correct": right_only,
                "both_not_correct": len(expected_ids) - both_correct - left_only - right_only,
                "mcnemar_exact_p_value": p_value,
            })

    return {
        "sample_ids": expected_ids,
        "runs": rows,
        "pairwise": pairwise,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_paths", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()

    payload = compare(args.result_paths)
    for row in payload["runs"]:
        format_rate = row["format_compliance_rate"]
        format_text = "n/a" if format_rate is None else f"{format_rate:.2%}"
        print(
            f"{row['condition']} | {row['model']} | {row['prompt_name']} | n={row['num_samples']} | "
            f"accuracy={row['accuracy']:.2%} | invalid={row['invalid_rate']:.2%} | "
            f"format={format_text}"
        )

    if args.output:
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Comparison saved to: {args.output}")


if __name__ == "__main__":
    main()
